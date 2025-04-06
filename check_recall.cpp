#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <future>
#include <vector>
#include <iomanip>
#include <map> 

#include "utils.hpp"

struct ThreadResult {
    float total_recall = 0.0f;
    size_t valid_entries = 0;
    std::unordered_map<size_t, float> batch_recall_sum;
    std::unordered_map<size_t, size_t> batch_entry_count;
};

ThreadResult process_chunk(const std::vector<SearchResult<uint32_t>>& res,
                          const std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_set<uint32_t>>>& gt_map,
                          size_t start, size_t end) {
    ThreadResult result;
    result.batch_recall_sum.reserve(end - start);
    result.batch_entry_count.reserve(end - start);

    for (size_t i = start; i < end; ++i) {
        const auto& res_entry = res[i];
        size_t offset = res_entry.insert_offset;
        size_t qidx = res_entry.query_idx;

        if (gt_map.count(offset) == 0 || gt_map.at(offset).count(qidx) == 0) {
            std::cerr << "Warning: No matching ground truth for offset " << offset 
                      << ", query " << qidx << std::endl;
            continue;
        }

        const auto& gt_tags = gt_map.at(offset).at(qidx);
        const auto& res_tags = res_entry.tags;
        size_t tags_size = res_tags.size();
        if (tags_size == 0) continue;

        size_t hits = 0;
        for (uint32_t tag : res_tags) {
            if (gt_tags.count(tag) > 0) {
                hits++;
            }
        }

        float recall = static_cast<float>(hits) / tags_size;
        result.total_recall += recall;
        result.valid_entries++;

        result.batch_recall_sum[offset] += recall;
        result.batch_entry_count[offset]++;
    }

    return result;
}

float check_recall(std::vector<SearchResult<uint32_t>>& res, std::vector<SearchResult<uint32_t>>& gt, const std::string& recall_path) {
    std::unordered_map<size_t, std::unordered_map<size_t, std::unordered_set<uint32_t>>> gt_map;
    gt_map.reserve(gt.size());
    for (const auto& gt_entry : gt) {
        gt_map[gt_entry.insert_offset][gt_entry.query_idx].insert(gt_entry.tags.begin(), gt_entry.tags.end());
    }

    size_t thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 1;
    size_t chunk_size = res.size() / thread_count;
    if (chunk_size == 0) chunk_size = 1;

    std::vector<std::future<ThreadResult>> futures;
    for (size_t i = 0; i < res.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, res.size());
        futures.push_back(std::async(std::launch::async, process_chunk, std::cref(res), std::cref(gt_map), i, end));
    }

    float total_recall = 0.0f;
    size_t valid_entries = 0;
    std::unordered_map<size_t, float> batch_recall_sum;
    std::unordered_map<size_t, size_t> batch_entry_count;
    batch_recall_sum.reserve(res.size());
    batch_entry_count.reserve(res.size());

    size_t total_chunks = futures.size();
    size_t completed_chunks = 0;

    for (auto& f : futures) {
        ThreadResult result = f.get();
        total_recall += result.total_recall;
        valid_entries += result.valid_entries;

        for (const auto& [offset, recall_sum] : result.batch_recall_sum) {
            batch_recall_sum[offset] += recall_sum;
        }
        for (const auto& [offset, count] : result.batch_entry_count) {
            batch_entry_count[offset] += count;
        }

        completed_chunks++;
        float progress = static_cast<float>(completed_chunks) / total_chunks * 100.0f;
        std::cout << "Progress: " << completed_chunks << "/" << total_chunks 
                  << " chunks completed (" << std::fixed << std::setprecision(2) << progress << "%)" << std::endl;
    }

    if (valid_entries == 0) {
        std::cerr << "Error: No valid entries to compute recall" << std::endl;
        return 0.0f;
    }

    float average_recall = total_recall / valid_entries;

    std::map<size_t, float> sorted_batch_recall_sum(batch_recall_sum.begin(), batch_recall_sum.end());
    std::map<size_t, size_t> sorted_batch_entry_count(batch_entry_count.begin(), batch_entry_count.end());

    std::stringstream ss;
    ss << "Batch Offset\tAverage Recall\tEntry Count\n";
    for (const auto& [offset, recall_sum] : sorted_batch_recall_sum) {
        float batch_avg_recall = recall_sum / sorted_batch_entry_count[offset];
        ss << offset << "\t" << batch_avg_recall << "\t" << sorted_batch_entry_count[offset] << "\n";
        std::cout << "Batch " << offset << ": Average recall = " << batch_avg_recall 
                  << " (" << sorted_batch_entry_count[offset] << " entries)" << std::endl;
    }

    std::ofstream out_file(recall_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Failed to open recall output file: " << recall_path << std::endl;
    } else {
        out_file << ss.str();
        out_file.close();
    }

    std::cout << "Computed recall for " << valid_entries << " entries, average recall: " 
              << average_recall << std::endl;
    return average_recall;
}

int main(int argc, char* argv[]) {
    std::string res_path, gt_path, recall_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--res_path" && i + 1 < argc) {
            res_path = argv[++i];
        } else if (arg == "--gt_path" && i + 1 < argc) {
            gt_path = argv[++i];
        } else if (arg == "--recall_path" && i + 1 < argc) {
            recall_path = argv[++i];
        }
    }

    if (res_path.empty() || gt_path.empty() || recall_path.empty()) {
        std::cerr << "Error: Missing --res_path, --gt_path, or --recall_path" << std::endl;
        return 1;
    }

    try {
        std::vector<SearchResult<uint32_t>> res, gt; 
        read_results(res, res_path);
        read_results(gt, gt_path);

        float recall = check_recall(res, gt, recall_path);
        std::cout << "Final average recall: " << recall << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}