export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
go build -o bench main.go
./bench