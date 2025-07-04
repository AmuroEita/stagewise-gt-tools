#pragma once

#include <omp.h>
#include <stdint.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

struct QParams {
    size_t ef_search;
    size_t beam_width;
    float alpha;
    size_t visit_limit;

    QParams() = default;

    QParams(size_t ef_search) : ef_search(ef_search) {}

    QParams(size_t ef_search, size_t beam_width, float alpha,
            size_t visit_limit)
        : ef_search(ef_search),
          beam_width(beam_width),
          alpha(alpha),
          visit_limit(visit_limit) {}
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
   public:
    virtual ~IndexBase() = default;
    virtual void build(const T* data, const TagT* tags, size_t num_points) = 0;
    virtual int insert(const T* point, const TagT tag) = 0;
    virtual int batch_insert(const T* batch_data, const TagT* batch_tags,
                             size_t num_points) = 0;
    virtual void set_query_params(const QParams& params) = 0;
    virtual int search(const T* query, size_t k,
                       std::vector<TagT>& res_tags) = 0;

    virtual int batch_search(const T* batch_queries, uint32_t k,
                             size_t num_queries, TagT** batch_results) = 0;

    virtual void save_stat(const std::string& filename) {}
};