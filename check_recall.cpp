#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.hpp"

struct ThreadResult {
    float total_recall = 0.0f;
    size_t valid_entries = 0;
    std::unordered_map<size_t, float> batch_recall_sum;
    std::unordered_map<size_t, size_t> batch_entry_count;
};

ThreadResult process_chunk(
    const std::vector<SearchResult<uint32_t>>& res,
    const std::unordered_map<
        size_t, std::unordered_map<size_t, std::unordered_set<uint32_t>>>&
        gt_map,
    size_t start, size_t end) {
    ThreadResult result{0.0f, 0, {}, {}};
    for (size_t i = start; i < end; ++i) {
        const auto& res_entry = res[i];
        const auto& gt_tags =
            gt_map.at(res_entry.insert_offset).at(res_entry.query_idx);

        size_t matches = 0;
        for (auto tag : res_entry.tags) {
            if (gt_tags.count(tag)) matches++;
        }
        float recall = static_cast<float>(matches) / res_entry.tags.size();
        result.total_recall += recall;
        result.valid_entries++;
        result.batch_recall_sum[res_entry.insert_offset] += recall;
        result.batch_entry_count[res_entry.insert_offset]++;
    }
    return result;
}

float check_recall(std::vector<SearchResult<uint32_t>>& res,
                   std::vector<SearchResult<uint32_t>>& gt,
                   const std::string& recall_path) {
    std::unordered_map<size_t,
                       std::unordered_map<size_t, std::unordered_set<uint32_t>>>
        gt_map;
    gt_map.reserve(gt.size());
    size_t ties_detected = 0;

    for (const auto& gt_entry : gt) {
        auto& tag_set = gt_map[gt_entry.insert_offset][gt_entry.query_idx];
        tag_set.insert(gt_entry.tags.begin(), gt_entry.tags.end());

        std::vector<std::pair<uint32_t, float>> tagged_distances;
        for (size_t i = 0; i < gt_entry.tags.size(); ++i) {
            tagged_distances.emplace_back(gt_entry.tags[i],
                                          gt_entry.distances[i]);
        }
        std::sort(
            tagged_distances.begin(), tagged_distances.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        for (size_t i = 1; i < tagged_distances.size(); ++i) {
            if (std::abs(tagged_distances[i].second -
                         tagged_distances[i - 1].second) < 1e-6) {
                tag_set.insert(tagged_distances[i].first);
                tag_set.insert(tagged_distances[i - 1].first);
                ties_detected++;
            }
        }
    }

    size_t thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 1;
    size_t chunk_size = res.size() / thread_count;
    if (chunk_size == 0) chunk_size = 1;

    std::vector<std::future<ThreadResult>> futures;
    for (size_t i = 0; i < res.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, res.size());
        futures.push_back(std::async(std::launch::async, process_chunk,
                                     std::cref(res), std::cref(gt_map), i,
                                     end));
    }

    float total_recall = 0.0f;
    size_t valid_entries = 0;
    std::unordered_map<size_t, float> batch_recall_sum;
    std::unordered_map<size_t, size_t> batch_entry_count;

    size_t total_chunks = futures.size();
    size_t completed_chunks = 0;

    for (auto& f : futures) {
        try {
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
            float progress =
                static_cast<float>(completed_chunks) / total_chunks * 100.0f;
            std::cout << "Progress: " << completed_chunks << "/" << total_chunks
                      << " (" << std::fixed << std::setprecision(2) << progress
                      << "%)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Thread exception: " << e.what() << std::endl;
        }
    }

    if (valid_entries == 0) {
        std::cerr << "Error: No valid entries to compute recall" << std::endl;
        return 0.0f;
    }

    float average_recall = total_recall / valid_entries;
    std::cout << "Detected " << ties_detected
              << " tie instances in ground truth" << std::endl;

    std::map<size_t, float> sorted_batch_recall_sum(batch_recall_sum.begin(),
                                                    batch_recall_sum.end());
    std::map<size_t, size_t> sorted_batch_entry_count(batch_entry_count.begin(),
                                                      batch_entry_count.end());

    std::stringstream ss;
    ss << "Batch Offset\tAverage Recall\tEntry Count\n";
    for (const auto& [offset, recall_sum] : sorted_batch_recall_sum) {
        float batch_avg_recall = recall_sum / sorted_batch_entry_count[offset];
        ss << offset << "\t" << batch_avg_recall << "\t"
           << sorted_batch_entry_count[offset] << "\n";
        std::cout << "Batch " << offset
                  << ": Average recall = " << batch_avg_recall << " ("
                  << sorted_batch_entry_count[offset] << " queries)"
                  << std::endl;
    }

    std::ofstream out_file(recall_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Failed to open recall output file: " << recall_path
                  << std::endl;
    } else {
        out_file << ss.str();
        out_file.close();
    }

    std::cout << "Computed recall for " << valid_entries
              << " queries, average stage-wise recall: " << average_recall << std::endl;
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
        std::cerr << "Error: Missing --res_path, --gt_path, or --recall_path"
                  << std::endl;
        return 1;
    }

    try {
        using TagT = uint32_t;

        std::vector<SearchResult<TagT>> res, gt;
        read_results(res, res_path);

        load_gt<TagT>(gt, gt_path);

        float recall = check_recall(res, gt, recall_path);
        std::cout << "Final average recall: " << recall << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}