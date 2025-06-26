export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
go build -o bench main.go
./bench -config config/parlay_hnsw/sift_b100_w50_t16.yaml