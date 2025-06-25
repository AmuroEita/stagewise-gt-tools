#pragma once

#include "../index.hpp"
#include "parlayann/algorithms/HNSW/HNSW.hpp"
#include "parlayann/algorithms/HNSW/dist.hpp"
#include "parlayann/algorithms/HNSW/type_point.hpp"
#include "parlayann/algorithms/utils/euclidian_point.h"
#include "parlayann/algorithms/utils/point_range.h"
#include <memory>
#include <vector>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayHNSW : public IndexBase<T, TagT, LabelT> {
   public:
    using Point = T;
    using TagType = TagT;
    using LabelType = LabelT;
    using DistanceType = float;
    
    using desc = descr_l2<T>;

    ParlayHNSW(size_t dim, size_t max_elements, size_t M, size_t ef_construction, float m_l, float alpha)
        : dim_(dim),
          graph_degree_(M), 
          ef_construction_(ef_construction), 
          m_l_(m_l), 
          alpha_(alpha) {
    }

    void build(T *data, size_t num_points, std::vector<TagT> &tags) {
        using PointType = parlayANN::Euclidian_Point<T>;
        
        auto ps = parlay::delayed_seq<PointType>(num_points, [&](size_t i) {
            return PointType(data + i * dim_);
        });
        
        index_ = std::make_unique<ANN::HNSW<desc>>(
            ps.begin(), ps.end(), dim_, m_l_, graph_degree_, ef_construction_, alpha_
        );
    }

    int batch_insert(T *data, size_t num_points, std::vector<TagT> &tags) override {
        if (!index_) return -1;
        using PointType = parlayANN::Euclidian_Point<T>;
        auto ps = parlay::delayed_seq<PointType>(num_points, [&](size_t i) {
            return PointType(data + i * dim_);
        });
        uint32_t start_id = tags.empty() ? 0 : tags[0];
        index_->batch_insert(ps.begin(), ps.end(), start_id);
        return 0;
    }

    int insert_point(T *point, const TagT &tag) override {
        std::cerr << "ParlayHNSW does not support dynamic single insertion" << std::endl;
        return -1;
    }

    void set_query_params(const size_t Ls) override {
        ef_search_ = Ls;
    }

    void search_with_tags(const T *query, size_t k, size_t Ls,
                          std::vector<TagT> &result_tags) override {
        if (!index_) {
            return;
        }
        
        parlayANN::Euclidian_Point<T> query_point(query);
        
        auto results = index_->search(query_point, k, Ls);
        
        result_tags.clear();
        result_tags.reserve(results.size());
        for (const auto& result : results) {
            result_tags.push_back(static_cast<TagT>(result.first));
        }
    }

   private:
    size_t dim_;
    uint32_t graph_degree_; // M
    uint32_t ef_construction_;
    uint32_t ef_search_;
    float m_l_;
    float alpha_;
    std::unique_ptr<ANN::HNSW<desc>> index_;
};