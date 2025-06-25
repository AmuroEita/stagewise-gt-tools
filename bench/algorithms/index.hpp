#pragma once

#include <omp.h>
#include <stdint.h>

#include <chrono>
#include <cstdint>
#include <vector>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
   public:
    std::vector<double> insert_latencies;
    std::vector<double> search_latencies;

    virtual ~IndexBase() = default;
    virtual void build(const T* data, const TagT* tags, size_t num_points) = 0;
    virtual int insert(const T* point, const TagT tag) = 0;
    virtual int batch_insert(const T* batch_data, const TagT* batch_tags,
                             size_t num_points) = 0;
    virtual void set_query_params(size_t Ls) = 0;
    virtual int search_with_tags(const T* query, size_t k, size_t Ls,
                                 std::vector<TagT>& res_tags) = 0;

    virtual int batch_search(const T* batch_queries, uint32_t k, uint32_t Ls,
                             size_t num_queries, TagT** batch_results) = 0;

    void get_insert_latencies(std::vector<double>& times) {
        times = insert_latencies;
    }

    void get_search_latencies(std::vector<double>& times) {
        times = search_latencies;
    }
};