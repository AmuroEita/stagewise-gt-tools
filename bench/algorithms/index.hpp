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
    virtual void set_query_params(const size_t Ls) = 0;
    virtual void search_with_tags(const T *query, size_t k, size_t Ls,
                                  std::vector<TagT> &res_tags) = 0;

    int batch_insert(const std::vector<T *> &batch_data,
                     const std::vector<TagT> &batch_tags,
                     uint64_t batch_id = 0) {
        int success_count = 0;
        insert_times.resize(batch_data.size(), 0.0);
#pragma omp parallel for reduction(+ : success_count)
        for (size_t i = 0; i < batch_data.size(); ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            if (batch_id == 0) {
                if (insert_point(batch_data[i], batch_tags[i]) == 0) {
                    success_count++;
                }
            } else {
                if (insert_point(batch_data[i], batch_tags[i], batch_id) == 0) {
                    success_count++;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            insert_times[i] =
                std::chrono::duration<double, std::micro>(end - start).count();
        }
        return success_count == batch_data.size() ? 0 : -1;
    }

    void batch_search(const std::vector<T *> &batch_queries, uint32_t k,
                      uint32_t Ls,
                      std::vector<std::vector<TagT>> &batch_results,
                      uint64_t batch_id = 0) {
        batch_results.resize(batch_queries.size());
        search_times.resize(batch_queries.size(), 0.0);
#pragma omp parallel for
        for (size_t i = 0; i < batch_queries.size(); ++i) {
            batch_results[i].reserve(k);
            auto start = std::chrono::high_resolution_clock::now();
            if (batch_id == 0) {
                search_with_tags(batch_queries[i], k, Ls, batch_results[i]);
            } else {
                search_with_tags(batch_queries[i], k, Ls, batch_results[i],
                                 batch_id);
            }
            auto end = std::chrono::high_resolution_clock::now();
            search_times[i] =
                std::chrono::duration<double, std::micro>(end - start).count();
        }
    }

    void get_insert_times(std::vector<double> &times) { times = insert_times; }

    void get_search_times(std::vector<double> &times) { times = search_times; }
};