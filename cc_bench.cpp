#include <getopt.h>

#include <cstring>
#include <iostream>
#include <string>

#include "algorithms/hnsw.hpp"
#include "bench.hpp"

int main(int argc, char *argv[]) {
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        std::cerr << "PAPI library init error: " << PAPI_strerror(retval)
                  << " (retval=" << retval << ")" << std::endl;
        exit(1);
    }

    std::string data_type, data_path, query_path, batch_res_path, gt_path;
    size_t begin_num = 5000, batch_size = 100;
    float write_ratio = 0.5;
    size_t recall_at = 10, R = 16, Ls = 50,
           num_threads = std::thread::hardware_concurrency();

    struct option long_options[] = {{"data_type", required_argument, 0, 0},
                                    {"data_path", required_argument, 0, 0},
                                    {"query_path", required_argument, 0, 0},
                                    {"batch_res_path", required_argument, 0, 0},
                                    {"begin_num", required_argument, 0, 0},
                                    {"max_elements", required_argument, 0, 0},
                                    {"write_ratio", required_argument, 0, 0},
                                    {"batch_size", required_argument, 0, 0},
                                    {"recall_at", required_argument, 0, 0},
                                    {"R", required_argument, 0, 0},
                                    {"Ls", required_argument, 0, 0},
                                    {"dim", required_argument, 0, 0},
                                    {"num_threads", required_argument, 0, 0},
                                    {"gt_path", required_argument, 0, 0},
                                    {0, 0, 0, 0}};

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "", long_options, &option_index)) !=
           -1) {
        if (c == 0) {
            std::string opt_name = long_options[option_index].name;
            if (opt_name == "data_type")
                data_type = optarg;
            else if (opt_name == "data_path")
                data_path = optarg;
            else if (opt_name == "query_path")
                query_path = optarg;
            else if (opt_name == "batch_res_path")
                batch_res_path = optarg;
            else if (opt_name == "begin_num")
                begin_num = std::stoul(optarg);
            else if (opt_name == "write_ratio")
                write_ratio = std::stof(optarg);
            else if (opt_name == "batch_size")
                batch_size = std::stoul(optarg);
            else if (opt_name == "recall_at")
                recall_at = std::stoul(optarg);
            else if (opt_name == "R")
                R = std::stoul(optarg);
            else if (opt_name == "Ls")
                Ls = std::stoul(optarg);
            else if (opt_name == "num_threads")
                num_threads = std::stoul(optarg);
            else if (opt_name == "gt_path")
                gt_path = optarg;
        }
    }

    if (data_path.empty() || query_path.empty() || batch_res_path.empty() ||
        gt_path.empty()) {
        std::cerr << "Need --data_path, --query_path, --batch_res_path and "
                     "--gt_path\n";
        return 1;
    }

    using TagT = uint32_t;
    using LabelT = uint32_t;
    std::vector<SearchResult<TagT>> search_results;

    size_t data_num, data_dim, aligned_dim;
    get_bin_metadata(data_path, data_num, data_dim);
    search_results.reserve(data_num * (1 / write_ratio - 1));

    if (data_type == "float") {
        using IndexType = HNSW<float, TagT, LabelT>;
        std::unique_ptr<IndexBase<float, TagT, LabelT>> index(
            new IndexType(data_dim, data_num, R, Ls));
        measure_performance(
            [&]() {
                concurrent_bench<float, TagT, LabelT>(
                    data_path, query_path, begin_num, write_ratio, batch_size,
                    recall_at, Ls, num_threads, std::move(index),
                    search_results);
            },
            true);

        overall_recall<float, TagT, LabelT>(query_path, recall_at, Ls,
                                            std::move(index), gt_path);
    } else if (data_type == "int8_t") {
    } else if (data_type == "uint8_t") {
    } else {
        std::cerr << "Unknown data type: " << data_type << "\n";
        return 1;
    }

    write_results(search_results, batch_res_path);

    return 0;
}