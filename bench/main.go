package main

import (
	"ANN-CC-bench/bench/internal"
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"golang.org/x/time/rate"
	"gopkg.in/yaml.v3"
)

type TaskType int

const (
	InsertTask TaskType = iota
	SearchTask
)

type Task struct {
	Type      TaskType
	Data      [][]float32
	Tags      []uint32
	QueryIdx  uint32
	RecallAt  uint32
	Ls        uint32
	BatchId   uint64
	Timestamp time.Time
}

type Stat struct {
	InsertQPS         float64
	SearchQPS         float64
	MeanInsertLatency float64
	MeanSearchLatency float64
}

type Index interface {
	BatchInsert(data [][]float32, tags []uint32, batchId uint64) error
	BatchSearch(queries [][]float32, recallAt, Ls uint, batchId uint64) ([][]uint32, error)
}

type Bench struct {
	taskQueue       chan Task
	index           Index
	stats           Stat
	mu              sync.Mutex
	rwMu            sync.RWMutex
	wg              sync.WaitGroup
	insertCnt       int
	searchCnt       int
	insertLatencies []float64
	searchLatencies []float64
	rateLimiter     *rate.Limiter
	searchResults   []*internal.SearchResult
	resultsMu       sync.Mutex
	config          *Config
}

func ConcurrentBench(index Index, config Config) *Bench {
	return &Bench{
		taskQueue:       make(chan Task, config.Workload.QueueSize),
		index:           index,
		insertLatencies: make([]float64, 0),
		searchLatencies: make([]float64, 0),
		rateLimiter:     rate.NewLimiter(rate.Limit(config.Workload.InputRate), int(config.Workload.InputRate)),
		config:          &config,
	}
}

func (b *Bench) ProduceTasks(data []float32, queries []float32, dim int, config *Config) {
	defer close(b.taskQueue)

	beginNum := config.Data.BeginNum
	writeRatio := config.Workload.WriteRatio
	batchSize := config.Data.BatchSize

	insertTotal := len(data)
	searchTotal := int(float64(insertTotal) * (1 - writeRatio) / writeRatio)
	searchBatchSize := int(float64(batchSize) * (1 - writeRatio) / writeRatio)

	b.insertLatencies = make([]float64, insertTotal)
	b.searchLatencies = make([]float64, searchTotal)

	startInsertOffset := beginNum
	startSearchOffset := 0
	if config.Workload.QueryNewData {
		startSearchOffset = beginNum
	}
	queryIdx := 0
	totalQueries := len(queries)
	
	var insertBatchId uint64 = 0
	var searchBatchId uint64 = 0

	for startInsertOffset < insertTotal || startSearchOffset < searchTotal {
		endInsertOffset := min(startInsertOffset+batchSize, insertTotal)
		if startInsertOffset < endInsertOffset {
			batchData := make([][]float32, 0, batchSize)
			batchTags := make([]uint32, 0, batchSize)
			for i := startInsertOffset; i < endInsertOffset; i++ {
				start := i * dim
				end := start + dim
				batchData = append(batchData, data[start:end])
				batchTags = append(batchTags, uint32(beginNum+i))
			}

			if err := b.rateLimiter.Wait(context.Background()); err != nil {
				fmt.Printf("Rate limit error: %v\n", err)
				continue
			}

			b.taskQueue <- Task{
				Type:      InsertTask,
				Data:      batchData,
				Tags:      batchTags,
				BatchId:   insertBatchId,
				Timestamp: time.Now(),
			}
			insertBatchId++
			startInsertOffset = endInsertOffset
			b.insertCnt += len(batchData)
		}

		endSearchOffset := min(startSearchOffset+searchBatchSize, searchTotal)
		batchQueries := make([][]float32, 0, searchBatchSize)
		batchTags := make([]uint32, 0, searchBatchSize)

		if config.Workload.QueryNewData {
			for i := startSearchOffset; i < endSearchOffset; i++ {
				randomIdx := startInsertOffset - (i - startSearchOffset + 1)
				if randomIdx < 0 {
					randomIdx = 0
				}
				start := randomIdx * dim
				end := start + dim
				batchQueries = append(batchQueries, data[start:end])
				batchTags = append(batchTags, uint32(beginNum+randomIdx))
			}
		} else {
			for i := startSearchOffset; i < endSearchOffset; i++ {
				queryIdx = (queryIdx + 1) % totalQueries
				start := queryIdx * dim
				end := start + dim
				batchQueries = append(batchQueries, queries[start:end])
				batchTags = append(batchTags, uint32(queryIdx))
			}
		}

		if err := b.rateLimiter.Wait(context.Background()); err != nil {
			fmt.Printf("Rate limit error: %v\n", err)
			continue
		}

		b.taskQueue <- Task{
			Type:      SearchTask,
			Data:      batchQueries,
			Tags:      batchTags,
			RecallAt:  config.Search.RecallAt,
			Ls:        config.Search.Ls,
			BatchId:   searchBatchId,
			Timestamp: time.Now(),
		}
		searchBatchId++
		startSearchOffset = endSearchOffset
		b.searchCnt += len(batchQueries)
	}
}

func (b *Bench) ConsumeTasks(numWorkers int) {
	for i := 0; i < numWorkers; i++ {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			for task := range b.taskQueue {
				start := time.Now()
				switch task.Type {
				case InsertTask:
					if !b.config.Index.UseBatchId {
						b.rwMu.Lock()
					}
					err := b.index.BatchInsert(task.Data, task.Tags, task.BatchId)
					if !b.config.Index.UseBatchId {
						b.rwMu.Unlock()
					}
					if err != nil {
						fmt.Printf("Insert error: %v\n", err)
						continue
					}
					b.insertLatencies[b.insertCnt] = float64(time.Since(start).Microseconds())
					b.insertCnt++
				case SearchTask:
					if !b.config.Index.UseBatchId {
						b.rwMu.RLock()
					}
					results, err := b.index.BatchSearch(task.Data, uint(task.RecallAt), uint(task.Ls), task.BatchId)
					if !b.config.Index.UseBatchId {
						b.rwMu.RUnlock()
					}
					if err != nil {
						fmt.Printf("Search error: %v\n", err)
						continue
					}
					b.resultsMu.Lock()
					for i, tags := range results {
						result := internal.NewSearchResult(
							uint64(b.insertCnt),
							uint64(i),
							tags,
						)
						b.searchResults = append(b.searchResults, result)
					}
					b.resultsMu.Unlock()
					b.searchLatencies[b.searchCnt] = float64(time.Since(start).Microseconds())
					b.searchCnt++
				}
			}
		}()
	}
}

func (b *Bench) CollectStats(elapsedSec float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.insertCnt > 0 {
		b.stats.InsertQPS = float64(b.insertCnt) / elapsedSec
		b.stats.MeanInsertLatency = mean(b.insertLatencies)
	}
	if b.searchCnt > 0 {
		b.stats.SearchQPS = float64(b.searchCnt) / elapsedSec
		b.stats.MeanSearchLatency = mean(b.searchLatencies)
	}

	fmt.Printf("Insert QPS: %.2f, Mean Insert Latency: %.2f us\n", b.stats.InsertQPS, b.stats.MeanInsertLatency)
	fmt.Printf("Search QPS: %.2f, Mean Search Latency: %.2f us\n", b.stats.SearchQPS, b.stats.MeanSearchLatency)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

type Config struct {
	Data struct {
		DatasetName string `yaml:"dataset_name"`
		MaxElements uint64 `yaml:"max_elements"`
		BeginNum    int    `yaml:"begin_num"`
		BatchSize   int    `yaml:"batch_size"`
		MaxQueries  int    `yaml:"max_queries"`
		DataType    string `yaml:"data_type"`
		DataPath    string `yaml:"data_path"`
		QueryPath   string `yaml:"query_path"`
	} `yaml:"data"`

	Index struct {
		IndexType string `yaml:"index_type"`
		M         int    `yaml:"m"`
		Lb        int    `yaml:"lb"`
		UseBatchId bool  `yaml:"use_batch_id"`
	} `yaml:"index"`

	Search struct {
		RecallAt uint32 `yaml:"recall_at"`
		Ls       uint32 `yaml:"ls"`
	} `yaml:"search"`

	Workload struct {
		WriteRatio   float64 `yaml:"write_ratio"`
		NumThreads   int     `yaml:"num_threads"`
		QueueSize    int     `yaml:"queue_size"`
		QueryNewData bool    `yaml:"query_new_data"`
		InputRate    float64 `yaml:"input_rate"`
		Async        bool    `yaml:"async"`
	} `yaml:"workload"`

	Result struct {
		OutputDir    string `yaml:"output_dir"`
		GtPath       string `yaml:"gt_path"`
		BatchResPath string `yaml:"batch_res_path"`
	} `yaml:"result"`
}

func loadConfig(filename string) (*Config, error) {
	buf, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %v", err)
	}

	config := &Config{}
	err = yaml.Unmarshal(buf, config)
	if err != nil {
		return nil, fmt.Errorf("error parsing config file: %v", err)
	}

	return config, nil
}

func main() {
	config, err := loadConfig("config/config.yaml")
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		return
	}

	var dataNum uint64
	var dataDim int
	internal.GetBinMetadata(config.Data.DataPath, &dataNum, &dataDim)

	data, _, _, _, err := internal.LoadAlignedBin(config.Data.DataPath)
	if err != nil {
		fmt.Printf("Failed to load data: %v\n", err)
		return
	}

	queries, _, _, _, err := internal.LoadAlignedBin(config.Data.QueryPath)
	if err != nil {
		fmt.Printf("Failed to load queries: %v\n", err)
		return
	}

	var index Index
	switch config.Index.IndexType {
	case "hnsw":
		params := internal.IndexParams{
			Dim:         dataDim,
			MaxElements: config.Data.MaxElements,
			M:           config.Index.M,
			Lb:          config.Index.Lb,
			DataType:    internal.DataTypeFloat,
		}
		index = internal.NewIndex(internal.IndexTypeHNSW, params)
	default:
		fmt.Printf("Unsupported index type: %s\n", config.Index.IndexType)
		return
	}

	beginNum := config.Data.BeginNum
	if beginNum != 0 {
		preData := make([][]float32, 0, beginNum)
		preTags := make([]uint32, 0, beginNum)
		for i := 0; i < beginNum; i++ {
			start := i * dataDim
			end := start + dataDim
			preData = append(preData, data[start:end])
			preTags = append(preTags, uint32(i))
		}
		if err := index.BatchInsert(preData, preTags, 0); err != nil {
			fmt.Printf("Warn start error: %v\n", err)
			return
		}
	}

	var bench *Bench
	bench = ConcurrentBench(index, *config)
	bench.searchResults = make([]*internal.SearchResult, 0, config.Data.MaxElements)

	start := time.Now()

	go bench.ProduceTasks(data, queries, dataDim, config)
	bench.ConsumeTasks(config.Workload.NumThreads)

	bench.wg.Wait()
	elapsedSec := time.Since(start).Seconds()

	bench.CollectStats(elapsedSec)
}
