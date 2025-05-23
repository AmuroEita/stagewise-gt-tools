#pragma once

#include "index.hpp"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_VAMANA = 1,
    INDEX_TYPE_PARLAYANN = 2,
    INDEX_TYPE_CCHNSW = 3,
} IndexType;

void* create_index(IndexType type);

int build_index(void* index, float* data, size_t num_points, uint32_t* tags);

int insert_point(void* index, float* point, uint32_t tag);

void set_query_params(void* index, size_t Ls);

int search_with_tags(void* index, float* query, size_t k, size_t Ls, uint32_t* res_tags);

int batch_insert(void* index, float** batch_data, uint32_t* batch_tags, size_t batch_size);

int batch_search(void* index, float** batch_queries, size_t num_queries, uint32_t k, uint32_t Ls, uint32_t** batch_results);

#ifdef __cplusplus
}
#endif 