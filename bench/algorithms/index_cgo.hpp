#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INDEX_TYPE_HNSW = 0,
    INDEX_TYPE_PARLAYHNSW = 1,
    INDEX_TYPE_CCHNSW = 2,
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
} IndexParams;

void* create_index(IndexType type, IndexParams params);
void destroy_index();

int build_index(float* data, size_t num_points, uint32_t* tags);
int insert_point(float* point, uint32_t tag);
void set_query_params(size_t Ls);
int search_with_tags(float* query, size_t k, size_t Ls,
                     uint32_t* res_tags);
int batch_insert(float* batch_data, uint32_t* batch_tags, size_t batch_size);
int batch_search(float** batch_queries, size_t num_queries,
                 uint32_t k, uint32_t Ls, uint32_t** batch_results);

#ifdef __cplusplus
}
#endif