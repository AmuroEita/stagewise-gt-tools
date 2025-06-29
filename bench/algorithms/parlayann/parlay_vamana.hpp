#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "../index.hpp"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"
#include "parlayann/algorithms/utils/types.h"
#include "parlayann/algorithms/vamana/vamana.hpp"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayVamana : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = parlayANN::Euclidian_Point<float>;
    using Range = parlayANN::PointRange<Point>;
    using desc = parlayANN::Desc_HNSW<T, Point>;
    using BuildParams = parlayANN::BuildParams;
    using Graph = parlayANN::Graph<TagT>;
    using KnnIndex = parlayANN::knn_index<Range, Range, TagT>;
    using QueryParams = parlayANN::QueryParams;

    ParlayVamana(size_t dim, size_t max_elements, size_t M,
                 size_t ef_construction, float m_l, float alpha, bool two_pass,
                 size_t num_threads)
        : dim_(dim),
          graph_degree_(M),
          ef_construction_(ef_construction),
          m_l_(m_l),
          alpha_(alpha),
          two_pass_(two_pass),
          num_threads_(num_threads),
          max_elements_(max_elements),
          total_points_(0) {
        data_.resize(max_elements * dim_);
        setenv("PARLAY_NUM_THREADS", std::to_string(num_threads).c_str(), 1);
    }

    void build(const T* data, const TagT* tags, size_t num_points) override {
        data_.assign(data, data + num_points * dim_);
        total_points_ = num_points;
        Range points(data_.data(), total_points_, dim_);
        parlayANN::stats<TagT> build_stats(points->size());
        G_ = std::make_unique<Graph>(graph_degree_, max_elements_);
        BuildParams BP(graph_degree_, ef_construction_, alpha_, two_pass_ ? 2 : 1);
        index_ = std::make_unique<KnnIndex>(BP);
        index_->build_index(*G_, points, points, build_stats);                                
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) {
        size_t start_idx = G_->size();
        
        parlay::sequence<indexType> points = parlay::tabulate(
            num_points, 
            [&](size_t i) { return static_cast<indexType>(start_idx + i); }
        );
        
        return index_->incr_batch_insert(points, *G_, Range, Range, BuildStats, BP.alpha);
    }

    int insert(const T* point, const TagT tag) override {
        std::cerr << "ParlayHNSW does not support dynamic single insertion"
                  << std::endl;
        return -1;
    }

    void set_query_params(const QParams& params) override {
        query_params_ = params;
    }

    int search(const T* query, size_t k, const QParams& params,
               std::vector<TagT>& result_tags) override {
        std::cerr << "ParlayHNSW does not support dynamic single search"
                  << std::endl;
        return -1;
    }

    int batch_search(const T* batch_queries, uint32_t k, size_t num_queries,
                     TagT** batch_results) {
        QueryParams QP(k, BP.L, 1.35, (long)Points.size(), (long)G.max_degree());
        
        parlay::parallel_for(0, num_queries, [&](size_t i) {
            Point qpoint(reinterpret_cast<const float*>(batch_queries + i * dim_), dim_);

            auto search_results = beam_search_rerank__<Point, QPoint, PR, QPR, indexType>(
                qpoint, qpoint, G_, Range, Range, 0, QP);
            
            auto& results = search_results.first;
            for (uint32_t j = 0; j < k && j < results.size(); j++) {
                batch_results[i][j] = static_cast<TagT>(results[j].first);
            }
        });
        
        return 0;
    }

   private:
    std::mutex index_mutex;

    size_t dim_;
    uint32_t graph_degree_;  // M
    uint32_t ef_construction_;
    float m_l_;
    float alpha_;
    size_t num_threads_;
    size_t max_elements_;
    size_t total_points_;
    bool two_pass_;

    std::unique_ptr<KnnIndex> index_;
    std::unique_ptr<Graph> G_;

    std::vector<T> data_;
    QParams query_params_;
};