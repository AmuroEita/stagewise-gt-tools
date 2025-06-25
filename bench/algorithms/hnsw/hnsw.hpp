#pragma once

#include <cstddef>

#include "../index.hpp"
#include "hnswlib/hnswlib/hnswlib.h"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSW : public IndexBase<T, TagT, LabelT> {
   public:
    HNSW(size_t dim, size_t max_elements, size_t M, size_t ef_construction,
         size_t num_threads)
        : dim(dim), space(dim), num_threads_(num_threads) {
        index = new hnswlib::HierarchicalNSW<T>(&space, max_elements, M,
                                                ef_construction);
    }

    void build(T *data, size_t num_points, std::vector<TagT> &tags) {
        for (size_t i = 0; i < num_points; i++) {
            index->addPoint(data + i * dim, tags[i]);
        }
    }

    int insert_point(T *data, const TagT &tag) override {
        index->addPoint(data, tag);
        return 0;
    }

    int batch_insert(const std::vector<T *> &batch_data,
                     const std::vector<TagT> &batch_tags) override {
        if (batch_data.size() != batch_tags.size()) {
            return -1;
        }
        int success_count = 0;
#pragma omp parallel for reduction(+ : success_count) num_threads(num_threads_)
        for (size_t i = 0; i < batch_data.size(); ++i) {
            index->addPoint(batch_data[i], batch_tags[i]);
            success_count++;
        }
        return success_count == batch_data.size() ? 0 : -1;
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

    size_t num_threads_ = 1;
    size_t dim;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<T> *index;
    bool is_ef_set = false;
};
