#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0]
              << " <input.fvecs> <num_vectors> <output.fvecs>" << std::endl;
    return 1;
  }

  const char *input_file = argv[1];
  int n = std::atoi(argv[2]);
  const char *output_file = argv[3];

  if (n <= 0) {
    std::cerr << "Number of vectors must be positive" << std::endl;
    return 1;
  }

  auto [data, dim] = read_fvecs(input_file, n);
  std::cout << "Read " << n << " vectors with dimension " << dim << " from "
            << input_file << std::endl;

  write_fvecs(output_file, data, n, dim);
  std::cout << "Wrote " << n << " vectors to " << output_file << std::endl;

  return 0;
}