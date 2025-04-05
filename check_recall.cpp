#include "utils.hpp"

std::vector<SearchResult<uint32_t>> load_search_result(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
  }

  int k, b;  
  in.read(reinterpret_cast<char*>(&k), sizeof(int));  
  in.read(reinterpret_cast<char*>(&b), sizeof(int));  

  if (k <= 0 || b <= 0) {
      throw std::runtime_error("Invalid header values in bin file");
  }

  std::vector<SearchResult<uint32_t>> results;
  size_t total_entries = 0;  

  for (int batch_idx = 0; batch_idx < b; ++batch_idx) {
      int current_base_size;  
      int n;                  
      in.read(reinterpret_cast<char*>(&current_base_size), sizeof(int));
      in.read(reinterpret_cast<char*>(&n), sizeof(int));

      if (n <= 0) {
          throw std::runtime_error("Invalid number of queries in batch " + std::to_string(batch_idx));
      }

      std::vector<std::vector<uint32_t>> batch_ids(n, std::vector<uint32_t>(k));
      for (int i = 0; i < n; ++i) {
          for (int j = 0; j < k; ++j) {
              int id;
              in.read(reinterpret_cast<char*>(&id), sizeof(int));
              batch_ids[i][j] = static_cast<uint32_t>(id);
          }
      }

      for (int i = 0; i < n; ++i) {
          results.emplace_back(
              static_cast<size_t>(current_base_size),  
              static_cast<size_t>(i),                  
              batch_ids[i]                             
          );
      }
      total_entries += n;
  }

  in.close();
  std::cout << "Read " << total_entries << " SearchResult entries from " << filename 
            << " (k: " << k << ", batches: " << b << ")" << std::endl;

  return results;
}

float check_recall(std::vector<SearchResult<uint32_t>>& res, std::vector<SearchResult<uint32_t>>& gt) {
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<uint32_t>>> gt_map;
  for (const auto& gt_entry : gt) {
      gt_map[gt_entry.insert_offset][gt_entry.query_idx] = gt_entry.tags;
  }

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
  }

  if (valid_entries == 0) {
      std::cerr << "Error: No valid entries to compute recall" << std::endl;
      return 0.0f;
  }

  float average_recall = total_recall / valid_entries;
  std::cout << "Computed recall for " << valid_entries << " entries, average recall: " 
            << average_recall << std::endl;
  return average_recall;
}

int main(int argc, char* argv[]) {
  std::string res_path, gt_path;

  for (int i = 1; i < argc; ++i) {  
      std::string arg = argv[i];
      if (arg == "--res_path" && i + 1 < argc) {
          res_path = argv[++i];
      } else if (arg == "--gt_path" && i + 1 < argc) {
          gt_path = argv[++i];
      }
  }

  if (res_path.empty() || gt_path.empty()) {
      std::cerr << "Error: Missing --res_path or --gt_path" << std::endl;
      return 1;
  }

  try {
      std::vector<SearchResult<uint32_t>> res = load_search_result(res_path);
      std::vector<SearchResult<uint32_t>> gt = load_search_result(gt_path);

      float recall = check_recall(res, gt);
      std::cout << "Final average recall: " << recall << std::endl;
  } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
  }

  return 0;
}