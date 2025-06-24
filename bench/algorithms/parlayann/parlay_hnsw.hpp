#pragma once

#include "../index.hpp"
#include "parlay/"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class ParlayHNSW : public IndexBase<T, TagT, LabelT> {
   public:
    ParlayHNSW(size_t dim, size_t max_elements, size_t M, size_t ef_construction)
        : dim(dim), space(dim) {
        index = new hnswlib::HierarchicalNSW<T>(&space, max_elements, M,
                                                ef_construction);
    }

    void build(T *data, size_t num_points, std::vector<TagT> &tags) {
        
    }

    int insert_point(T *data, const TagT &tag) {
        index->addPoint(data, tag);
        return 0;
    }

    void set_query_params(const size_t Ls) override { index->setEf(Ls); }

    void search_with_tags(const T *query, size_t k, size_t Ls,
                          std::vector<TagT> &result_tags) override {
        if (!is_ef_set) {
            index->setEf(Ls);
            is_ef_set = true;
        }

        auto result = index->searchKnn(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
    }

    size_t dim;
    PointRange<Point> Points;
    ANN::HNSW<Desc_HNSW<T, Point>;> *index;
};