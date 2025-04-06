# Quick Start

## Build
```
mkdir build && cd build
```

## Dataset
```
mkdir data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```

## Run Compute GT
```

```

## Run Concurrent Bench
```
./cc_bench --data_type float --data_path ../data/sift/sift_base.fbin --query_path ../data/sift/sift_query_1k.fbin --begin_num 5000 --max_elements 1000000 --write_ratio 0.5 --batch_size 100 --recall_at 10 --R 32 --Ls 40 --dim 128 --num_threads 64 --res_path sift_base_100.res
```