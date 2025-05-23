#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

std::pair<std::vector<float>, int> read_fvecs(const char *filename, int n) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open input file");
        exit(-1);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        exit(-1);
    }

    char *fileptr =
        static_cast<char *>(mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (fileptr == MAP_FAILED) {
        perror("mmap");
        exit(-1);
    }
    close(fd);

    int dim = *reinterpret_cast<int *>(fileptr);
    size_t vector_size_bytes = sizeof(int) + dim * sizeof(float);
    size_t total_vectors = sb.st_size / vector_size_bytes;

    if (n > total_vectors) {
        std::cerr << "Requested " << n << " vectors, but file only has "
                  << total_vectors << std::endl;
        n = total_vectors;
    }

    std::vector<float> data;
    data.reserve(n * dim);

    char *ptr = fileptr;
    for (int i = 0; i < n; ++i) {
        int current_dim = *reinterpret_cast<int *>(ptr);
        if (current_dim != dim) {
            std::cerr << "Dimension mismatch at vector " << i << std::endl;
            break;
        }
        ptr += sizeof(int);
        float *vec = reinterpret_cast<float *>(ptr);
        data.insert(data.end(), vec, vec + dim);
        ptr += dim * sizeof(float);
    }

    munmap(fileptr, sb.st_size);
    return {data, dim};
}

void write_fvecs(const char *filename, const std::vector<float> &data, int n,
                 int dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        perror("open output file");
        exit(-1);
    }

    for (int i = 0; i < n; ++i) {
        out.write(reinterpret_cast<const char *>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char *>(data.data() + i * dim),
                  dim * sizeof(float));
    }
    out.close();
}

std::unordered_set<int> read_hotspot_ids(const char *filename) {
    std::unordered_set<int> ids;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open hotspots file: " << filename << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(file, line)) {
        // 读取每行的第一个数字（query id）
        int id = std::stoi(line.substr(0, line.find(' ')));
        ids.insert(id);
    }
    return ids;
}

std::pair<std::vector<float>, int> extract_vectors_by_ids(
    const std::vector<float> &all_data, const std::unordered_set<int> &ids,
    int dim) {
    std::vector<float> selected_data;
    selected_data.reserve(ids.size() * dim);

    for (int id : ids) {
        if (id >= 0 && id * dim < all_data.size()) {
            selected_data.insert(selected_data.end(),
                                 all_data.begin() + id * dim,
                                 all_data.begin() + (id + 1) * dim);
        }
    }

    return {selected_data, dim};
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout
            << "Usage: " << argv[0]
            << " <input.fvecs> <num_vectors> <hotspots.txt> <output.fvecs>"
            << std::endl;
        return 1;
    }

    const char *input_file = argv[1];
    int n = std::atoi(argv[2]);
    const char *hotspots_file = argv[3];
    const char *output_file = argv[4];

    if (n <= 0) {
        std::cerr << "Number of vectors must be positive" << std::endl;
        return 1;
    }

    auto [data, dim] = read_fvecs(input_file, n);
    std::cout << "Read " << n << " vectors with dimension " << dim << " from "
              << input_file << std::endl;

    auto hotspot_ids = read_hotspot_ids(hotspots_file);
    std::cout << "Read " << hotspot_ids.size() << " hotspot IDs from "
              << hotspots_file << std::endl;

    auto [selected_data, selected_dim] =
        extract_vectors_by_ids(data, hotspot_ids, dim);
    std::cout << "Extracted " << selected_data.size() / dim << " vectors"
              << std::endl;

    write_fvecs(output_file, selected_data, selected_data.size() / dim, dim);
    std::cout << "Wrote " << selected_data.size() / dim << " vectors to "
              << output_file << std::endl;

    return 0;
}