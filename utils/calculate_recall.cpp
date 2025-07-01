// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "disk_utils.h"
#include "utils.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << argv[0] << " <ground_truth_bin> <our_results_bin>  <r> "
                  << std::endl;
        return -1;
    }
    uint32_t *gold_std = NULL;
    float *gs_dist = nullptr;
    uint32_t *our_results = NULL;
    float *or_dist = nullptr;
    size_t points_num, points_num_gs, points_num_or;
    size_t dim_gs;
    size_t dim_or;
    diskann::load_truthset(argv[1], gold_std, gs_dist, points_num_gs, dim_gs);
    diskann::load_truthset(argv[2], our_results, or_dist, points_num_or,
                           dim_or);

    if (points_num_gs != points_num_or) {
        std::cout << "Error. Number of queries mismatch in ground truth and "
                     "our results"
                  << std::endl;
        return -1;
    }
    points_num = points_num_gs;

    uint32_t recall_at = std::atoi(argv[3]);

    if ((dim_or < recall_at) || (recall_at > dim_gs)) {
        std::cout << "ground truth has size " << dim_gs << "; our set has "
                  << dim_or << " points. Asking for recall " << recall_at
                  << std::endl;
        return -1;
    }
    std::cout << "Calculating recall@" << recall_at << std::endl;
    double recall_val = diskann::calculate_recall(
        (uint32_t)points_num, gold_std, gs_dist, (uint32_t)dim_gs, our_results,
        (uint32_t)dim_or, (uint32_t)recall_at);

    //  double avg_recall = (recall*1.0)/(points_num*1.0);
    std::cout << "Avg. recall@" << recall_at << " is " << recall_val << "\n";
}  // Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "disk_utils.h"
#include "utils.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << argv[0] << " <ground_truth_bin> <our_results_bin>  <r> "
                  << std::endl;
        return -1;
    }
    uint32_t *gold_std = NULL;
    float *gs_dist = nullptr;
    uint32_t *our_results = NULL;
    float *or_dist = nullptr;
    size_t points_num, points_num_gs, points_num_or;
    size_t dim_gs;
    size_t dim_or;
    diskann::load_truthset(argv[1], gold_std, gs_dist, points_num_gs, dim_gs);
    diskann::load_truthset(argv[2], our_results, or_dist, points_num_or,
                           dim_or);

    if (points_num_gs != points_num_or) {
        std::cout << "Error. Number of queries mismatch in ground truth and "
                     "our results"
                  << std::endl;
        return -1;
    }
    points_num = points_num_gs;

    uint32_t recall_at = std::atoi(argv[3]);

    if ((dim_or < recall_at) || (recall_at > dim_gs)) {
        std::cout << "ground truth has size " << dim_gs << "; our set has "
                  << dim_or << " points. Asking for recall " << recall_at
                  << std::endl;
        return -1;
    }
    std::cout << "Calculating recall@" << recall_at << std::endl;
    double recall_val = diskann::calculate_recall(
        (uint32_t)points_num, gold_std, gs_dist, (uint32_t)dim_gs, our_results,
        (uint32_t)dim_or, (uint32_t)recall_at);

    //  double avg_recall = (recall*1.0)/(points_num*1.0);
    std::cout << "Avg. recall@" << recall_at << " is " << recall_val << "\n";
}

double calculate_recall(uint32_t num_queries, uint32_t *gold_std,
                        float *gs_dist, uint32_t dim_gs, uint32_t *our_results,
                        uint32_t dim_or, uint32_t recall_at) {
    double total_recall = 0;
    std::set<uint32_t> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        if (gs_dist != nullptr) {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec,
                   res_vec + recall_at);  // change to recall_at for recall k@k
                                          // or dim_or for k@dim_or
        uint32_t cur_recall = 0;
        for (auto &v : gt) {
            if (res.find(v) != res.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
}

inline void load_truthset(const std::string &bin_file, uint32_t *&ids,
                          float *&dists, size_t &npts, size_t &dim) {
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    diskann::cout << "Reading truthset file " << bin_file.c_str() << " ..."
                  << std::endl;
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    diskann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim
                  << "... " << std::endl;

    int truthset_type = -1;  // 1 means truthset has ids and distances, 2 means
                             // only ids, -1 is error
    size_t expected_file_size_with_dists =
        2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists) truthset_type = 1;

    size_t expected_file_size_just_ids =
        npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids) truthset_type = 2;

    if (truthset_type == -1) {
        std::stringstream stream;
        stream
            << "Error. File size mismatch. File should have bin format, with "
               "npts followed by ngt followed by npts*ngt ids and optionally "
               "followed by npts*ngt distance values; actual size: "
            << actual_file_size
            << ", expected: " << expected_file_size_with_dists << " or "
            << expected_file_size_just_ids;
        diskann::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *)ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1) {
        dists = new float[npts * dim];
        reader.read((char *)dists, npts * dim * sizeof(float));
    }
}

double calculate_recall(uint32_t num_queries, uint32_t *gold_std,
                        float *gs_dist, uint32_t dim_gs, uint32_t *our_results,
                        uint32_t dim_or, uint32_t recall_at,
                        const tsl::robin_set<uint32_t> &active_tags) {
    double total_recall = 0;
    std::set<uint32_t> gt, res;
    bool printed = false;
    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        uint32_t active_points_count = 0;
        uint32_t cur_counter = 0;
        while (active_points_count < recall_at && cur_counter < dim_gs) {
            if (active_tags.find(*(gt_vec + cur_counter)) !=
                active_tags.end()) {
                active_points_count++;
            }
            cur_counter++;
        }
        if (active_tags.empty()) cur_counter = recall_at;

        if ((active_points_count < recall_at && !active_tags.empty()) &&
            !printed) {
            diskann::cout << "Warning: Couldn't find enough closest neighbors "
                          << active_points_count << "/" << recall_at
                          << " from "
                             "truthset for query # "
                          << i
                          << ". Will result in under-reported value of recall."
                          << std::endl;
            printed = true;
        }
        if (gs_dist != nullptr) {
            tie_breaker = cur_counter - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec, res_vec + recall_at);
        uint32_t cur_recall = 0;
        for (auto &v : res) {
            if (gt.find(v) != gt.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return ((double)(total_recall / (num_queries))) *
           ((double)(100.0 / recall_at));
}

double calculate_recall(uint32_t num_queries, uint32_t *gold_std,
                        float *gs_dist, uint32_t dim_gs, uint32_t *our_results,
                        uint32_t dim_or, uint32_t recall_at) {
    double total_recall = 0;
    std::set<uint32_t> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        if (gs_dist != nullptr) {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec,
                   res_vec + recall_at);  // change to recall_at for recall k@k
                                          // or dim_or for k@dim_or
        uint32_t cur_recall = 0;
        for (auto &v : gt) {
            if (res.find(v) != res.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
}

double calculate_recall(uint32_t num_queries, uint32_t *gold_std,
                        float *gs_dist, uint32_t dim_gs, uint32_t *our_results,
                        uint32_t dim_or, uint32_t recall_at,
                        const tsl::robin_set<uint32_t> &active_tags) {
    double total_recall = 0;
    std::set<uint32_t> gt, res;
    bool printed = false;
    for (size_t i = 0; i < num_queries; i++) {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        uint32_t active_points_count = 0;
        uint32_t cur_counter = 0;
        while (active_points_count < recall_at && cur_counter < dim_gs) {
            if (active_tags.find(*(gt_vec + cur_counter)) !=
                active_tags.end()) {
                active_points_count++;
            }
            cur_counter++;
        }
        if (active_tags.empty()) cur_counter = recall_at;

        if ((active_points_count < recall_at && !active_tags.empty()) &&
            !printed) {
            diskann::cout << "Warning: Couldn't find enough closest neighbors "
                          << active_points_count << "/" << recall_at
                          << " from "
                             "truthset for query # "
                          << i
                          << ". Will result in under-reported value of recall."
                          << std::endl;
            printed = true;
        }
        if (gs_dist != nullptr) {
            tie_breaker = cur_counter - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs &&
                   gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec, res_vec + recall_at);
        uint32_t cur_recall = 0;
        for (auto &v : res) {
            if (gt.find(v) != gt.end()) {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return ((double)(total_recall / (num_queries))) *
           ((double)(100.0 / recall_at));
}