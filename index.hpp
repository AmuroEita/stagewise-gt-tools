#pragma once

#include <cstdint>
#include <omp.h>
#include <vector>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
   public:
    virtual ~IndexBase() = default;
    virtual void build(T *data, size_t num_points, std::vector<TagT> &tags) = 0;
    virtual int insert_point(T *point, const TagT &tag) = 0;
    virtual void set_query_params(const size_t Ls) = 0;
    virtual void search_with_tags(const T *query, size_t k, size_t Ls,
                                  std::vector<TagT> &res_tags) = 0;

    int batch_insert(const std::vector<T*>& batch_data, const std::vector<TagT>& batch_tags) {
        int success_count = 0;
#pragma omp parallel for reduction(+:success_count)
        for (size_t i = 0; i < batch_data.size(); ++i) {
            if (insert_point(batch_data[i], batch_tags[i]) == 0) {
                success_count++;
            }
        }
        return success_count == batch_data.size() ? 0 : -1;
    }
    
    void batch_search(const std::vector<T*>& batch_queries, uint32_t k, uint32_t Ls, 
                     std::vector<std::vector<TagT>>& batch_results) {
        batch_results.resize(batch_queries.size());
#pragma omp parallel for
        for (size_t i = 0; i < batch_queries.size(); ++i) {
            batch_results[i].reserve(k);
            search_with_tags(batch_queries[i], k, Ls, batch_results[i]);
        }
    }
};