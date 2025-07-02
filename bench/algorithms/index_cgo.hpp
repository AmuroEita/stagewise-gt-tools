#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_PARLAYHNSW = 1,
    INDEX_TYPE_VAMANA = 2,
    INDEX_TYPE_PARLAYVAMANA = 3,
    INDEX_TYPE_CCHNSW = 4,
} IndexType;

typedef enum {
    DATA_TYPE_FLOAT = 0,
    DATA_TYPE_INT8 = 1,
    DATA_TYPE_UINT8 = 2,
} DataType;

typedef struct {
    size_t dim;
    size_t max_elements;
    size_t M;
    size_t ef_construction;
    float level_m;
    float alpha;
    size_t visit_limit;
    size_t num_threads;
    DataType data_type;
} IndexParams;

typedef struct {
    size_t ef_search;
    size_t beam_width;
    float alpha;
    size_t visit_limit;
} C_QueryParams;

void* create_index(IndexType type, IndexParams params);
void destroy_index(void* index_ptr);

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points);
int insert(void* index_ptr, float* point, uint32_t tag);
void set_query_params(void* index_ptr, C_QueryParams params);
int search(void* index_ptr, float* query, size_t k, uint32_t* res_tags);
int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size);
int batch_search(void* index_ptr, float* batch_queries, uint32_t k,
                 size_t num_queries, uint32_t** batch_results);

#ifdef __cplusplus
}
#endif