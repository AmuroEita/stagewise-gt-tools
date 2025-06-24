# ANN-CC-bench

A benchmark tool for ANN (Approximate Nearest Neighbor) algorithms.

## Quick Start with Docker

### Build Image

```bash
docker build -t ann-cc-bench .
```

### Run Container

Basic usage:
```bash
docker run -it --rm ann-cc-bench
```

To mount a data directory:
```bash
docker run -it --rm -v /path/to/your/data:/app/data ann-cc-bench
```

## Configuration

The benchmark configuration file is located at `bench/config/config.yaml`. Modify the parameters according to your needs:

- Dataset paths
- Algorithm parameters
- Benchmark settings

## Built-in Algorithms

- HNSW(hnswlib) : https://github.com/Kwan-Yiu/HNSW-CC-Bench