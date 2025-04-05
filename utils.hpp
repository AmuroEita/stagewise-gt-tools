#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstring>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
public:
    virtual ~IndexBase() = default;
    void build(T* data, size_t num_points, const std::vector<TagT>& tags) {

    }

    int insert_point(T* point, TagT tag) { 
        return 0; 
    }

    void search_with_tags(const T* query, size_t k, size_t Ls, TagT* tags, std::vector<T*>& res) {
        
    }
};

template <typename TagT>
struct SearchResult {
    size_t insert_offset;
    size_t query_idx;
    std::vector<TagT> tags;
    SearchResult(size_t offset, size_t idx, const std::vector<TagT>& t) 
        : insert_offset(offset), query_idx(idx), tags(t) {}
};

void read_results(std::vector<SearchResult<uint32_t>>& res, const std::string& res_path) {
  res.clear();
  
  std::ifstream in_file(res_path);
  if (!in_file.is_open()) {
      throw std::runtime_error("Unable to open file: " + res_path);
  }

  std::string line;
  size_t current_offset = 0;
  bool first_batch = true;

  while (std::getline(in_file, line)) {
      if (line.empty()) {
          continue;
      }

      if (line == "batch") {
          if (!first_batch) {
              current_offset++;  
          }
          first_batch = false;
          continue;
      }

      size_t query_idx;
      std::istringstream iss(line);
      if (!(iss >> query_idx)) {
          in_file.close();
          throw std::runtime_error("Invalid query_idx format at offset " + 
                                 std::to_string(current_offset));
      }

      if (!std::getline(in_file, line)) {
          in_file.close();
          throw std::runtime_error("Missing tags line at offset " + 
                                 std::to_string(current_offset));
      }

      std::vector<uint32_t> tags;
      std::istringstream tags_stream(line);
      uint32_t tag;
      while (tags_stream >> tag) {
          tags.push_back(tag);
      }

      res.emplace_back(current_offset, query_idx, tags);
  }

  in_file.close();
}

void write_results(std::vector<SearchResult<uint32_t>>& res, const std::string& res_path) {
  std::sort(res.begin(), res.end(), 
      [](const SearchResult<uint32_t>& a, const SearchResult<uint32_t>& b) {
          return a.insert_offset < b.insert_offset;
      });

  std::ofstream out_file(res_path);
  if (!out_file.is_open()) {
      throw std::runtime_error("Unable to open file: " + res_path);
  }

  size_t current_offset = res.empty() ? 0 : res[0].insert_offset;
  bool first_batch = true;

  for (const auto& result : res) {
      if (result.insert_offset != current_offset) {
          current_offset = result.insert_offset;
          out_file << "\nbatch\n";
      }
      else if (first_batch) {
          out_file << "batch\n";
          first_batch = false;
      }

      out_file << result.query_idx << "\n";
      
      for (size_t i = 0; i < result.tags.size(); ++i) {
          out_file << result.tags[i];
          if (i < result.tags.size() - 1) {
              out_file << " ";  
          }
      }
      out_file << "\n";
  }

  out_file.close();
}

template <typename T>
inline void load_aligned_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t &rounded_dim)
{
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        std::cout << "Reading bin file " << bin_file << " ..." << std::flush;
        reader.open(bin_file, std::ios::binary | std::ios::ate);

        uint64_t actual_file_size = reader.tellg();
        reader.seekg(0);

        int npts_i32, dim_i32;
        reader.read((char *)&npts_i32, sizeof(int));
        reader.read((char *)&dim_i32, sizeof(int));
        npts = static_cast<size_t>(npts_i32);
        dim = static_cast<size_t>(dim_i32);

        size_t expected_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
        if (actual_file_size != expected_file_size)
        {
            std::stringstream stream;
            stream << "Error: File size mismatch. Actual size is " << actual_file_size 
                   << ", expected size is " << expected_file_size 
                   << " (npts = " << npts << ", dim = " << dim << ", sizeof(T) = " << sizeof(T) << ")";
            throw std::runtime_error(stream.str());
        }

        rounded_dim = (dim + 7) & ~7; 
        std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim 
                  << ", aligned_dim = " << rounded_dim << "... " << std::flush;

        size_t alloc_size = npts * rounded_dim * sizeof(T);
        std::cout << "Allocating memory of " << alloc_size << " bytes... " << std::flush;
        data = new T[npts * rounded_dim]; 
        std::cout << "done. Copying data..." << std::flush;

        for (size_t i = 0; i < npts; i++)
        {
            reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
            std::memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
        }
        std::cout << " done." << std::endl;
    }
    catch (const std::ios_base::failure &e)
    {
        std::stringstream stream;
        stream << "Failed to read file " << bin_file << ": " << e.what();
        throw std::runtime_error(stream.str());
    }
    catch (const std::exception &e)
    {
        throw; 
    }
}

void get_bin_metadata(const std::string& filename, size_t& num_points, 
                      size_t& dimensions, size_t offset = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file.seekg(offset, std::ios::beg);
    if (file.fail()) {
        file.close();
        throw std::runtime_error("Failed to seek to offset: " + std::to_string(offset));
    }

    uint32_t metadata[2];
    file.read(reinterpret_cast<char*>(metadata), 2 * sizeof(uint32_t));
    if (file.gcount() != 2 * sizeof(uint32_t)) {
        file.close();
        throw std::runtime_error("Failed to read metadata at offset: " + std::to_string(offset));
    }

    num_points = metadata[0];
    dimensions = metadata[1];
    std::cout << "File " << filename << " contains " << num_points << " points and "
              << dimensions << " dimensions at offset " << offset << std::endl;

    file.close();
}