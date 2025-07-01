export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
go build -gcflags "all=-N -l" -o bench main.go
./bench -config config/hnsw/sift_b100_w50_t48.yaml