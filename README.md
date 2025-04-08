# Quick Start

## Build
```
mkdir build && cd build
```

## Prepare Datasets
```
mkdir data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
```

## Crop Query Datasets

```
```

## Run Compute Stage-wise GT
```
./compute_gt --base_path ../data/sift/sift_base.fvecs --query_path ../data/sift/sift_query_1k.fvecs --data_type float --k 20 --batch_gt_path test_batch.gt
```

## Run Concurrent Bench
```
./cc_bench --data_type float --data_path ../data/sift/sift_base.fbin --query_path ../data/sift/sift_query_1k.fbin --begin_num 5000 --write_ratio 0.5 --batch_size 100 --recall_at 10 --R 32 --Ls 100 --num_threads 64 --gt_path sift_full_recall.gt --batch_res_path sift_base_100.res 
```

## Check Stage-wise Recall
```
```