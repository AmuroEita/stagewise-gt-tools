#pragma once

#include <omp.h>
#include <stdint.h>

#include <chrono>
#include <cstdint>
#include <vector>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
    std::vector<double> insert_times;
    std::vector<double> search_times;

   public:
    virtual ~IndexBase() = default;
    virtual void build(T *data, size_t num_points, std::vector<TagT> &tags) = 0;
    virtual int insert_point(T *point, const TagT &tag) = 0;
    virtual int batch_insert(const std::vector<T *> &batch_data,
                             const std::vector<TagT> &batch_tags) = 0;
    virtual void set_query_params(const size_t Ls) = 0;
    virtual void search_with_tags(const T *query, size_t k, size_t Ls,
                                  std::vector<TagT> &res_tags) = 0;

    void batch_search(const std::vector<T *> &batch_queries, uint32_t k,
                      uint32_t Ls,
                      std::vector<std::vector<TagT>> &batch_results) = 0;

    void get_insert_times(std::vector<double> &times) { times = insert_times; }

    void get_search_times(std::vector<double> &times) { times = search_times; }
};