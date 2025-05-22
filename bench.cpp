#include <getopt.h>

#include <cstring>
#include <iostream>
#include <string>

#include "bench.hpp"
#include "algorithms/hnsw.hpp"
#include "utils.hpp"

void print_help() {
    std::cout
        << "HNSW-Bench Usage:\n"
        << "Usage: ./cc_bench [options]\n\n"
        << "Options:\n"
        << "  -h, --help                Show this help message\n"
        << "  -d, --dataset_name NAME   Name of the dataset\n"
        << "  -t, --data_type TYPE      Type of the data\n"
        << "  -p, --data_path PATH      Path to the data file\n"
        << "  -q, --query_path PATH     Path to the query file\n"
        << "  -b, --batch_res_path PATH Path to save batch results\n"
        << "  -i, --begin_num NUM       Initial number of points to build\n"
        << "  -m, --max_elements NUM    Maximum number of elements\n"
        << "  -w, --write_ratio RATIO   Write ratio (0-1)\n"
        << "  -s, --batch_size NUM      Batch size for processing\n"
        << "  -r, --recall_at NUM       k value for recall calculation\n"
        << "  -R, --R NUM               R parameter for index\n"
        << "  -L, --Lb NUM              Lb parameter for index\n"
        << "  -l, --Ls NUM              Ls parameter for search\n"
        << "  -n, --num_threads NUM     Number of threads\n"
        << "  -g, --gt_path PATH        Path to the ground truth file\n"
        << "  -o, --stat_path PATH      Path to save statistics\n"
        << "  -N, --query_new_data      Query new data\n"
        << "  -a, --async               Enable asynchronous processing\n\n"
        << "Example:\n"
        << "  ./bench -d sift -t float -p data.bin -q query.bin -b results/ "
           "-i 10000 "
        << "-w 0.5 -s 1000 -r 10 -R 16 -L 32 -l 100 -n 16 -g gt.bin -o "
           "stats.csv\n";
}

int main(int argc, char *argv[]) {
    if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 ||
                                    strcmp(argv[1], "--help") == 0))) {
        print_help();
        return 0;
    }

    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        std::cerr << "PAPI library init error: " << PAPI_strerror(retval)
                  << " (retval=" << retval << ")" << std::endl;
        exit(1);
    }

    std::string data_type, data_path, query_path, batch_res_path, gt_path,
        index_name, dataset_name, stat_path;
    size_t begin_num = 5000, batch_size = 100;
    float write_ratio = 0.5;
    size_t recall_at = 10, R = 16, Ls = 50, Lb = 50,
           num_threads = std::thread::hardware_concurrency();

    struct option long_options[] = {
        {"dataset_name", required_argument, 0, 'd'},
        {"data_type", required_argument, 0, 't'},
        {"data_path", required_argument, 0, 'p'},
        {"query_path", required_argument, 0, 'q'},
        {"batch_res_path", required_argument, 0, 'b'},
        {"begin_num", required_argument, 0, 'i'},
        {"max_elements", required_argument, 0, 'm'},
        {"write_ratio", required_argument, 0, 'w'},
        {"batch_size", required_argument, 0, 's'},
        {"recall_at", required_argument, 0, 'r'},
        {"R", required_argument, 0, 'R'},
        {"Lb", required_argument, 0, 'L'},
        {"Ls", required_argument, 0, 'l'},
        {"num_threads", required_argument, 0, 'n'},
        {"gt_path", required_argument, 0, 'g'},
        {"stat_path", required_argument, 0, 'o'},
        {"query_new_data", no_argument, 0, 'N'},
        {"async", no_argument, 0, 'a'},
        {0, 0, 0, 0}};

    int option_index = 0;
    int c;
    bool query_new_data = false;
    bool async = false;
    while ((c = getopt_long(argc, argv, "d:t:p:q:b:i:m:w:s:r:R:L:l:D:n:g:o:Na",
                            long_options, &option_index)) != -1) {
        switch (c) {
            case 'd':
                dataset_name = optarg;
                break;
            case 't':
                data_type = optarg;
                break;
            case 'p':
                data_path = optarg;
                break;
            case 'q':
                query_path = optarg;
                break;
            case 'b':
                batch_res_path = optarg;
                break;
            case 'i':
                begin_num = std::stoul(optarg);
                break;
            case 'm': /* max_elements */
                break;
            case 'w':
                write_ratio = std::stof(optarg);
                break;
            case 's':
                batch_size = std::stoul(optarg);
                break;
            case 'r':
                recall_at = std::stoul(optarg);
                break;
            case 'R':
                R = std::stoul(optarg);
                break;
            case 'L':
                Lb = std::stoul(optarg);
                break;
            case 'l':
                Ls = std::stoul(optarg);
                break;
            case 'n':
                num_threads = std::stoul(optarg);
                break;
            case 'g':
                gt_path = optarg;
                break;
            case 'o':
                stat_path = optarg;
                break;
            case 'N':
                query_new_data = true;
                break;
            case 'a':
                async = true;
                break;
            case 'h':
                print_help();
                return 0;
            case '?':
                return 1;
        }
    }

    using TagT = uint32_t;
    using LabelT = uint32_t;
    std::vector<SearchResult<TagT>> search_results;

    size_t data_num, data_dim, aligned_dim;
    get_bin_metadata(data_path, data_num, data_dim);
    search_results.reserve(data_num * (1 / write_ratio - 1));

    Stat stat("HNSW", dataset_name, R, Lb, Ls, write_ratio, num_threads,
              batch_size, batch_res_path);

    if (data_type == "float") {
        using IndexType = HNSW<float, TagT, LabelT>;
        std::unique_ptr<IndexBase<float, TagT, LabelT>> index(
            new IndexType(data_dim, data_num, R, Lb));

        measure_performance(
            [&]() {
                concurrent_bench<float, TagT, LabelT>(
                    data_path, query_path, begin_num, write_ratio, batch_size,
                    recall_at, Ls, num_threads, std::move(index),
                    search_results, stat, query_new_data, async);
            },
            true);

        if (query_new_data && recall_at == 1) {
            stagewise_recall<float, TagT, LabelT>(search_results, stat);
        } else {
            overall_recall<float, TagT, LabelT>(
                query_path, recall_at, Ls, std::move(index), gt_path, stat);
        }
    } else if (data_type == "int8_t") {
    } else if (data_type == "uint8_t") {
    } else {
        std::cerr << "Unknown data type: " << data_type << "\n";
        return 1;
    }

    save_stat(stat, stat_path);
    write_results(search_results, stat.stagewise_result_path);

    return 0;
}