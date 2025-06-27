#pragma once

#include <omp.h>
#include <stdint.h>

#include <chrono>
#include <cstdint>
#include <vector>
#include <iostream>

struct QParams {
    size_t Ls = 100;

    QParams() = default;
    QParams(size_t ls) : Ls(ls) {}

    size_t beam_width = 10;
    float alpha = 1.35;
    size_t visit_limit = 1000;

    QParams(size_t ls, size_t beam_w, float a, size_t visit_lim)
        : Ls(ls), beam_width(beam_w), alpha(a), visit_limit(visit_lim) {}
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
    virtual int search(const T* query, size_t k, const QParams& params,
                       std::vector<TagT>& res_tags) = 0;

    virtual int batch_search(const T* batch_queries, uint32_t k,
                             size_t num_queries, TagT** batch_results) = 0;
};