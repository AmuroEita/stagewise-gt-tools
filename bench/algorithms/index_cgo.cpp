#include "index_cgo.hpp"

#include <vector>

#include "hnsw/hnsw.hpp"
#include "index.hpp"
#include "parlayann/parlay_hnsw.hpp"

extern "C" {

static IndexBase<float>* g_index = nullptr;
static size_t g_dim = 0;

void* create_index(IndexType type, IndexParams params) {
    if (g_index) {
        delete g_index;
    }
    g_dim = params.dim;

    switch (type) {
        case INDEX_TYPE_HNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                g_index =
                    new HNSW<float>(params.dim, params.max_elements, params.M,
                                    params.Lb, params.num_threads);
                return g_index;
            }
            return nullptr;
        case INDEX_TYPE_PARLAYHNSW:
            if (params.data_type == DATA_TYPE_FLOAT) {
                g_index = new ParlayHNSW<float>(params.dim, params.max_elements,
                                                params.M, params.Lb, 1.15f,
                                                1.15f, params.num_threads);
                return g_index;
            }
            return nullptr;

        default:
            return nullptr;
    }
}

void destroy_index() {
    if (g_index) {
        delete g_index;
        g_index = nullptr;
        g_dim = 0;
    }
}

int build(float* data, uint32_t* tags, size_t num_points) {
    std::cout << "[build] num_points: " << num_points
              << ", g_dim: " << g_dim << std::endl;
    if (!g_index || !data || !tags) return -1;

    g_index->build(data, tags, num_points);
    return 0;
}

int insert(float* point, uint32_t tag) {
    if (!g_index || !point) return -1;

    return g_index->insert(point, tag);
}

void set_query_params(size_t Ls) {
    if (!g_index) return;

    g_index->set_query_params(Ls);
}

int search_with_tags(float* query, size_t k, size_t Ls, uint32_t* res_tags) {
    if (!g_index || !query || !res_tags) return -1;

    std::vector<uint32_t> results;
    g_index->search_with_tags(query, k, Ls, results);

    for (size_t i = 0; i < results.size(); ++i) {
        res_tags[i] = results[i];
    }
    return 0;
}

int batch_insert(float* batch_data, uint32_t* batch_tags, size_t batch_size) {
    if (!g_index || !batch_data || !batch_tags) return -1;
    return g_index->batch_insert(batch_data, batch_tags, batch_size);
}

int batch_search(float* batch_queries, uint32_t k,
                 uint32_t Ls, size_t num_queries, uint32_t** batch_results) {
    if (!g_index || !batch_queries) return -1;
    return g_index->batch_search(batch_queries, k, Ls, num_queries, batch_results);
}

}  // extern "C"