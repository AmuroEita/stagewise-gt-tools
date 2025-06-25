#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "../index.hpp"
#include "parlayann/algorithms/HNSW/HNSW.hpp"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"
#include "parlayann/algorithms/utils/types.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayHNSW : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = parlayANN::Euclidian_Point<float>;
    using Range = parlayANN::PointRange<Point>;
    using desc = parlayANN::Desc_HNSW<T, Point>;

    ParlayHNSW(size_t dim, size_t max_elements, size_t M,
               size_t ef_construction, float m_l, float alpha,
               size_t num_threads)
        : dim_(dim),
          graph_degree_(M),
          ef_construction_(ef_construction),
          m_l_(m_l),
          alpha_(alpha),
          num_threads_(num_threads) {}

    void build(const T* data, const TagT* tags, size_t num_points) override {
        Range points(data, num_points, dim_);
        auto ps = parlay::delayed_seq<Point>(
            points.size(), [&](size_t i) { return points[i]; });

        std::cout << "[build] start building HNSW index..." << std::endl;
        index_ = std::make_unique<ANN::HNSW<desc>>(ps.begin(), ps.end(), dim_,
                                                   m_l_, graph_degree_,
                                                   ef_construction_, alpha_);
        std::cout << "[build] finished building HNSW index." << std::endl;
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        if (!index_) return -1;
        uint32_t start_id = *batch_tags;

        Range points(batch_data, num_points, dim_);
        auto ps = parlay::delayed_seq<Point>(
            points.size(), [&](size_t i) { return points[i]; });

        index_->batch_insert(ps.begin(), ps.end(), start_id);
        return 0;
    }

    int insert(const T* point, const TagT tag) override {
        std::cerr << "ParlayHNSW does not support dynamic single insertion"
                  << std::endl;
        return -1;
    }

    void set_query_params(size_t Ls) override { ef_search_ = Ls; }

    int batch_search(const T* batch_queries, uint32_t k, uint32_t Ls,
                     size_t num_queries, TagT** batch_results) override {
        size_t beam_width = 10;
        float alpha = 1.35;
        size_t visit_limit = 1000;

        QueryParams QP(knn, beam_width, alpha, visit_limit,
                       std::min<int>(index_->max_degree(), 3 * visit_limit));

        Range qpoints(batch_queries, num_queries, dim_);
    }

    int search_with_tags(const T* query, size_t k, size_t Ls,
                         std::vector<TagT>& result_tags) override {
        std::cerr << "ParlayHNSW does not support dynamic single search"
                  << std::endl;
        return -1;
    }

   private:
    size_t dim_;
    uint32_t graph_degree_;  // M
    uint32_t ef_construction_;
    uint32_t ef_search_;
    float m_l_;
    float alpha_;
    size_t num_threads_;
    std::unique_ptr<ANN::HNSW<desc>> index_;
};