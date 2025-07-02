#include "index_cgo.hpp"

#include <cstdio>
#include <iostream>
#include <vector>

#include "hnsw/hnsw.hpp"
#include "index.hpp"
#include "parlayann/parlay_hnsw.hpp"
#include "parlayann/parlay_vamana.hpp"
#include "vamana/vamana.hpp"

extern "C" {

void* create_index(IndexType type, IndexParams params) {
    IndexBase<float>* index = nullptr;
    switch (type) {
        case INDEX_TYPE_HNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new HNSW<float>(params.max_elements, params.dim, 
                    params.num_threads, params.M, params.ef_construction);
            }
            break;
        case INDEX_TYPE_PARLAYHNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                index = new ParlayHNSW<float>(params.max_elements, params.dim,
                    params.num_threads, params.M, params.ef_construction,
                    params.level_m, params.alpha, params.visit_limit);
            }
            break;
        case INDEX_TYPE_VAMANA:
            // if (params.data_type == DATA_TYPE_FLOAT) {
            //     index = new Vamana<float>(params);
            // }
            // break;
        case INDEX_TYPE_PARLAYVAMANA:
            // if (params.data_type == DATA_TYPE_FLOAT) {
            //     index = new ParlayVamana<float>(params);
            // }
            // break;
        default:
            return nullptr;
    }
    return static_cast<void*>(index);
}

void destroy_index(void* index_ptr) {
    if (index_ptr) {
        delete static_cast<IndexBase<float>*>(index_ptr);
    }
}

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points) {
    if (!index_ptr || !data || !tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    index->build(data, tags, num_points);
    return 0;
}

int insert(void* index_ptr, float* point, uint32_t tag) {
    if (!index_ptr || !point) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->insert(point, tag);
}

void set_query_params(void* index_ptr, C_QueryParams params) {
    if (!index_ptr) return;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    QParams qparams(params.ef_search, params.beam_width, params.alpha,
                    params.visit_limit);
    index->set_query_params(qparams);
}

int search(void* index_ptr, float* query, size_t k, uint32_t* res_tags) {
    if (!index_ptr || !query || !res_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    std::vector<uint32_t> results;
    index->search(query, k, results);
    for (size_t i = 0; i < results.size(); ++i) {
        res_tags[i] = results[i];
    }
    return 0;
}

int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size) {
    if (!index_ptr || !batch_data || !batch_tags) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_insert(batch_data, batch_tags, batch_size);
}

int batch_search(void* index_ptr, float* batch_queries, uint32_t k,
                  size_t num_queries, uint32_t** batch_results) {
    if (!index_ptr || !batch_queries) return -1;
    auto index = static_cast<IndexBase<float>*>(index_ptr);
    return index->batch_search(batch_queries, k, num_queries, batch_results);
}

}  // extern "C"