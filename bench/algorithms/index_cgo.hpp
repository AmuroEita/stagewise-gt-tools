#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_PARLAYHNSW = 1,
    INDEX_TYPE_PARLAYVAMANA = 2,
    INDEX_TYPE_CCHNSW = 3,
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
    size_t Lb;
    DataType data_type;
    size_t num_threads;
} IndexParams;

typedef struct {
    size_t Ls;
    size_t beam_width;
    float alpha;
    size_t visit_limit;
} C_QParams;

void* create_index(IndexType type, IndexParams params);
void destroy_index(void* index_ptr);

int build(void* index_ptr, float* data, uint32_t* tags, size_t num_points);
int insert(void* index_ptr, float* point, uint32_t tag);
void set_query_params(void* index_ptr, C_QParams params);
int search(void* index_ptr, float* query, size_t k, C_QParams params,
           uint32_t* res_tags);
int batch_insert(void* index_ptr, float* batch_data, uint32_t* batch_tags,
                 size_t batch_size);
int batch_search(void* index_ptr, float* batch_queries, uint32_t k,
                 C_QParams params, size_t num_queries,
                 uint32_t** batch_results);

#ifdef __cplusplus
}
#endif