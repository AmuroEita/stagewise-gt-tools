#include "index_cgo.hpp"
#include "hnsw/hnsw.hpp"
#include <vector>

extern "C" {

void* create_index(IndexType type) {
    switch (type) {
        case INDEX_TYPE_HNSW:
            return new HNSW<float>();
        default:
            return nullptr;
    }
}

void destroy_index(void* index) {
    if (index) {
        delete static_cast<HNSW<float>*>(index);
    }
}

int build_index(void* index, float* data, size_t num_points, uint32_t* tags) {
    if (!index || !data || !tags) return -1;
    
    auto* idx = static_cast<HNSW<float>*>(index);
    std::vector<uint32_t> tags_vec(tags, tags + num_points);
    idx->build(data, num_points, tags_vec);
    return 0;
}

int insert_point(void* index, float* point, uint32_t tag) {
    if (!index || !point) return -1;
    
    auto* idx = static_cast<HNSW<float>*>(index);
    return idx->insert_point(point, tag);
}

void set_query_params(void* index, size_t Ls) {
    if (!index) return;
    
    auto* idx = static_cast<HNSW<float>*>(index);
    idx->set_query_params(Ls);
}

int search_with_tags(void* index, float* query, size_t k, size_t Ls, uint32_t* res_tags) {
    if (!index || !query || !res_tags) return -1;
    
    auto* idx = static_cast<HNSW<float>*>(index);
    std::vector<uint32_t> results;
    idx->search_with_tags(query, k, Ls, results);
    
    for (size_t i = 0; i < results.size(); ++i) {
        res_tags[i] = results[i];
    }
    return 0;
}

int batch_insert(void* index, float** batch_data, uint32_t* batch_tags, size_t batch_size) {
    if (!index || !batch_data || !batch_tags) return -1;
    
    auto* idx = static_cast<HNSW<float>*>(index);
    std::vector<float*> data_vec(batch_data, batch_data + batch_size);
    std::vector<uint32_t> tags_vec(batch_tags, batch_tags + batch_size);
    
    return idx->batch_insert(data_vec, tags_vec);
}

int batch_search(void* index, float** batch_queries, size_t num_queries, uint32_t k, uint32_t Ls, uint32_t** batch_results) {
    if (!index || !batch_queries || !batch_results) return -1;
    
    auto* idx = static_cast<HNSW<float>*>(index);
    std::vector<float*> queries_vec(batch_queries, batch_queries + num_queries);
    std::vector<std::vector<uint32_t>> results;
    
    idx->batch_search(queries_vec, k, Ls, results);
    
    for (size_t i = 0; i < results.size(); ++i) {
        batch_results[i] = new uint32_t[k];
        for (size_t j = 0; j < results[i].size(); ++j) {
            batch_results[i][j] = results[i][j];
        }
    }
    return 0;
}

} // extern "C" 