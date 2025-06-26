package main

import (
	"ANN-CC-bench/bench/internal"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/schollz/progressbar/v3"
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
	Timestamp time.Time
}

type Stat struct {
	InsertQPS         float64
	SearchQPS         float64
	MeanInsertLatency float64
	MeanSearchLatency float64
}

type Index interface {
	BatchInsert(data [][]float32, tags []uint32) error
	BatchSearch(queries [][]float32, recallAt, Ls uint) ([][]uint32, error)
	Build(data [][]float32, tags []uint32) error
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
		rateLimiter:     rate.NewLimiter(rate.Limit(config.Workload.InputRate*float64(config.Workload.NumThreads)), int(config.Workload.InputRate*float64(config.Workload.NumThreads))),
		config:          &config,
	}
}

func (b *Bench) ProduceTasks(data []float32, queries []float32, dim int, config *Config) {
	defer close(b.taskQueue)
	beginNum := config.Data.BeginNum
	writeRatio := config.Workload.WriteRatio
	batchSize := config.Data.BatchSize

	insertTotal := len(data) / dim
	searchTotal := config.Data.MaxQueries

	startInsertOffset := beginNum
	startSearchOffset := 0
	if config.Workload.QueryNewData {
		startSearchOffset = beginNum
	}
	queryIdx := 0
	totalQueries := len(queries) / dim

	bar := progressbar.NewOptions(
		int(insertTotal-beginNum),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowBytes(false),
		progressbar.OptionSetWidth(50),
		progressbar.OptionSetDescription("Insert: 0/s, Search: 0/s"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]=[reset]",
			SaucerHead:    "[green]>[reset]",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}),
	)

	startTime := time.Now()

	for startInsertOffset < insertTotal || startSearchOffset < searchTotal {
		insertBatchSize := int(float64(batchSize) * writeRatio)
		searchBatchSize := batchSize - insertBatchSize

		endInsertOffset := min(startInsertOffset+insertBatchSize, insertTotal)
		if endInsertOffset > startInsertOffset {
			task := Task{
				Type: InsertTask,
				Data: make([][]float32, 0, endInsertOffset-startInsertOffset),
				Tags: make([]uint32, 0, endInsertOffset-startInsertOffset),
			}
			for i := startInsertOffset; i < endInsertOffset; i++ {
				start := i * dim
				end := start + dim
				task.Data = append(task.Data, data[start:end])
				task.Tags = append(task.Tags, uint32(i))
			}
			b.taskQueue <- task
			bar.Add(endInsertOffset - startInsertOffset)
			startInsertOffset = endInsertOffset
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
			Timestamp: time.Now(),
		}
		startSearchOffset = endSearchOffset
		b.searchCnt += len(batchQueries)
		bar.Add(len(batchQueries))

		elapsed := time.Since(startTime).Seconds()
		if elapsed > 0 {
			insertQPS := float64(b.insertCnt) / elapsed
			searchQPS := float64(b.searchCnt) / elapsed
			bar.Describe(fmt.Sprintf("Insert: %.0f/s, Search: %.0f/s", insertQPS, searchQPS))
		}
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
					fmt.Println("xxxxx1.")
					if b.config.Workload.EnforceConsistency {
						b.rwMu.Lock()
					}
					fmt.Println("xxxxx2.")
					err := b.index.BatchInsert(task.Data, task.Tags)
					if b.config.Workload.EnforceConsistency {
						b.rwMu.Unlock()
					}
					if err != nil {
						fmt.Printf("Insert error: %v\n", err)
						continue
					}
					b.insertLatencies = append(b.insertLatencies, float64(time.Since(start).Microseconds()))
					b.insertCnt++
				case SearchTask:
					if b.config.Workload.EnforceConsistency {
						b.rwMu.RLock()
					}
					results, err := b.index.BatchSearch(task.Data, uint(task.RecallAt), uint(task.Ls))
					if b.config.Workload.EnforceConsistency {
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
					b.searchLatencies = append(b.searchLatencies, float64(time.Since(start).Microseconds()))
					b.searchCnt++
				}
			}
		}()
	}
}

func (b *Bench) CollectStats(elapsedSec float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	fmt.Println()

	if b.insertCnt > 0 {
		b.stats.InsertQPS = float64(b.insertCnt) / elapsedSec
		b.stats.MeanInsertLatency = mean(b.insertLatencies)
		p95 := percentile(b.insertLatencies, 0.95)
		p99 := percentile(b.insertLatencies, 0.99)
		fmt.Printf("Insert QPS: %.2f, Mean Insert Latency: %.2f us, P95: %.2f us, P99: %.2f us\n", b.stats.InsertQPS, b.stats.MeanInsertLatency, p95, p99)
	} else {
		fmt.Println("No insert operations performed.")
	}
	if b.searchCnt > 0 {
		b.stats.SearchQPS = float64(b.searchCnt) / elapsedSec
		b.stats.MeanSearchLatency = mean(b.searchLatencies)
		p95 := percentile(b.searchLatencies, 0.95)
		p99 := percentile(b.searchLatencies, 0.99)
		fmt.Printf("Search QPS: %.2f, Mean Search Latency: %.2f us, P95: %.2f us, P99: %.2f us\n", b.stats.SearchQPS, b.stats.MeanSearchLatency, p95, p99)
	} else {
		fmt.Println("No search operations performed.")
	}
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

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
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
	} `yaml:"index"`

	Search struct {
		RecallAt uint32 `yaml:"recall_at"`
		Ls       uint32 `yaml:"ls"`
	} `yaml:"search"`

	Workload struct {
		WriteRatio         float64 `yaml:"write_ratio"`
		NumThreads         int     `yaml:"num_threads"`
		QueueSize          int     `yaml:"queue_size"`
		QueryNewData       bool    `yaml:"query_new_data"`
		InputRate          float64 `yaml:"input_rate"`
		EnforceConsistency bool    `yaml:"enforce_consistency"`
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
	configPath := flag.String("config", "config/config.yaml", "配置文件路径")
	flag.Parse()

	config, err := loadConfig(*configPath)
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
	case "parlayhnsw":
		params := internal.IndexParams{
			Dim:         dataDim,
			MaxElements: config.Data.MaxElements,
			M:           config.Index.M,
			Lb:          config.Index.Lb,
			DataType:    internal.DataTypeFloat,
		}
		index = internal.NewIndex(internal.IndexTypeParlayHNSW, params)
	case "parlayvamana":
		params := internal.IndexParams{
			Dim:         dataDim,
			MaxElements: config.Data.MaxElements,
			M:           config.Index.M,
			Lb:          config.Index.Lb,
			DataType:    internal.DataTypeFloat,
		}
		index = internal.NewIndex(internal.IndexTypeParlayVamana, params)
	default:
		log.Fatalf("Unsupported index type: %s\n", config.Index.IndexType)
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
		fmt.Println("Begin data len:", len(preData))
		fmt.Println("Begin tags len:", len(preTags))
		if len(preData) > 0 {
			fmt.Println("Dim:", len(preData[0]))
		}
		if err := index.Build(preData, preTags); err != nil {
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
