#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "../index.hpp"
#include "parlayann/algorithms/vamana/vamana.hpp"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"
#include "parlayann/algorithms/utils/types.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayVamana : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = parlayANN::Euclidian_Point<float>;
    using Range = parlayANN::PointRange<Point>;
    using desc = parlayANN::Desc_HNSW<T, Point>;

    ParlayVamana(size_t dim, size_t max_elements, size_t M,
               size_t ef_construction, float m_l, float alpha,
               bool two_pass, size_t num_threads)
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
        Range points(data_.data(), total_points_, dim_);
        parlayANN::stats<TagT> build_stats(points->size());

        parlayANN::Graph<TagT> G = parlayANN::Graph<TagT>(graph_degree_, max_elements_);
        parlayANN::BuildParams BP(graph_degree_, ef_construction_, alpha_, two_pass ? 2 : 1);
        index_ = knn_index<Range, Range, TagT>(BP);
        index_->build_index(G, *points, *points, build_stats);                                
    }

    int batch_insert(const T* batch_data, const TagT* batch_tags,
                     size_t num_points) override {
        std::lock_guard<std::mutex> lock(index_mutex);

        assert((total_points_ + num_points) <= max_elements_);

        std::copy(batch_data, batch_data + num_points * dim_, 
                  data_.begin() + total_points_ * dim_);
        
        Range points(data_.data() + total_points_ * dim_, num_points, dim_);
        auto ps = parlay::delayed_seq<Point>(
            num_points, [&](size_t i) { return points[i]; });
        total_points_ += num_points;
        // index_->batch_insert(ps.begin(), ps.end(), batch_tags[0]);

        index_->batch_insert();
        return 0;
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
                     TagT** batch_results) override {
        parlayANN::QueryParams QP(k, query_params_.beam_width,
                                  query_params_.alpha,
                                  query_params_.visit_limit,
                                  std::min<int>(index_->get_threshold_m(0),
                                                3 * query_params_.visit_limit));
        Range qpoints(batch_queries, num_queries, dim_);
        parlay::sequence<TagT> starts(1, 0);

        for (size_t i = 0; i < num_queries; ++i) {
            batch_results[i] = new TagT[k];
        }

        Range points_range(data_.data(), total_points_, dim_);

        auto start = std::chrono::high_resolution_clock::now();
        auto graph = typename ANN::HNSW<desc>::graph(*index_, 0);
        parlay::parallel_for(0, num_queries, [&](size_t i) {
            auto q = qpoints[i];
            auto results = parlayANN::beam_search_impl<uint32_t>(
                q, graph, points_range, starts, QP);

            for (size_t j = 0; j < k && j < results.first.first.size(); ++j) {
                batch_results[i][j] = results.first.first[j].first;
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

    std::unique_ptr<knn_index<Range, Range, TagT>> index_;
    std::vector<T> data_;
    QParams query_params_;
};