export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
go build -gcflags "all=-N -l" -o bench main.go

for config_file in config/hnsw/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "=========================================="
        echo "Running benchmark with config: $config_file"
        echo "=========================================="
        ./bench -config "$config_file"
        echo ""
        echo "Benchmark completed for: $config_file"
        echo "=========================================="
        echo ""
    fi
done

echo "All benchmarks completed!"