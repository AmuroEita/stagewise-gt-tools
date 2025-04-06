#include <unordered_map>

#include "utils.hpp"

float check_recall(std::vector<SearchResult<uint32_t>>& res, std::vector<SearchResult<uint32_t>>& gt, const std::string& recall_path) {
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<uint32_t>>> gt_map;
    for (const auto& gt_entry : gt) {
        gt_map[gt_entry.insert_offset][gt_entry.query_idx] = gt_entry.tags;
    }

    std::unordered_map<size_t, float> batch_recall_sum;  
    std::unordered_map<size_t, size_t> batch_entry_count;  
    float total_recall = 0.0f;
    size_t valid_entries = 0;

    for (const auto& res_entry : res) {
        size_t offset = res_entry.insert_offset;  
        size_t qidx = res_entry.query_idx;

        if (gt_map.count(offset) == 0 || gt_map[offset].count(qidx) == 0) {
            std::cerr << "Warning: No matching ground truth for offset " << offset 
                      << ", query " << qidx << std::endl;
            continue;
        }

        const auto& gt_tags = gt_map[offset][qidx];
        const auto& res_tags = res_entry.tags;

        size_t hits = 0;
        for (uint32_t tag : res_tags) {
            if (std::find(gt_tags.begin(), gt_tags.end(), tag) != gt_tags.end()) {
                hits++;
            }
        }

        float recall = static_cast<float>(hits) / res_tags.size();
        total_recall += recall;
        valid_entries++;

        batch_recall_sum[offset] += recall;
        batch_entry_count[offset]++;
    }

    if (valid_entries == 0) {
        std::cerr << "Error: No valid entries to compute recall" << std::endl;
        return 0.0f;
    }

    float average_recall = total_recall / valid_entries;

    std::ofstream out_file(recall_path);
    if (!out_file.is_open()) {
        std::cerr << "Error: Failed to open recall output file: " << recall_path << std::endl;
    } else {
        out_file << "Batch Offset\tAverage Recall\tEntry Count\n";
        for (const auto& [offset, recall_sum] : batch_recall_sum) {
            float batch_avg_recall = recall_sum / batch_entry_count[offset];
            out_file << offset << "\t" << batch_avg_recall << "\t" << batch_entry_count[offset] << "\n";
            std::cout << "Batch " << offset << ": Average recall = " << batch_avg_recall 
                      << " (" << batch_entry_count[offset] << " entries)" << std::endl;
        }
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
        std::vector<SearchResult<uint32_t>> res = load_search_result(res_path);
        std::vector<SearchResult<uint32_t>> gt = load_search_result(gt_path);

        float recall = check_recall(res, gt, recall_path);
        std::cout << "Final average recall: " << recall << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}