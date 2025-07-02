package main

import (
	"ANN-CC-bench/bench/internal"
	"context"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/time/rate"
	"gopkg.in/yaml.v2"
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
	BatchSearch(queries [][]float32, recallAt uint32) ([][]uint32, error)
	Build(data [][]float32, tags []uint32) error
	SetQueryParams(params internal.QueryParams)
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
	insertPointCnt  int
	searchPointCnt  int
	globalInsertCnt int64
	startTime       time.Time
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
	beginNum := config.Data.BeginNum
	writeBatchSize := config.Data.WriteBatchSize
	writeRatio := config.Workload.WriteRatio
	searchBatchSize := int(float64(writeBatchSize) * (1.0/writeRatio - 1.0))

	insertTotal := len(data) / dim
	numInsertBatches := (insertTotal - beginNum + writeBatchSize - 1) / writeBatchSize

	// fmt.Printf("insertTotal=%d, max_elements=%d\n", insertTotal, config.Data.MaxElements)

	queryIdx := 0

	for batchIdx := 0; batchIdx < numInsertBatches; batchIdx++ {
		startInsertOffset := beginNum + batchIdx*writeBatchSize
		if startInsertOffset >= int(config.Data.MaxElements) {
			break
		}
		endInsertOffset := min(startInsertOffset+writeBatchSize, insertTotal)
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
		}

		batchQueries := make([][]float32, 0, searchBatchSize)
		batchTags := make([]uint32, 0, searchBatchSize)
		totalQueries := len(queries) / dim
		maxQueryIdx := min(config.Data.MaxQueries, totalQueries)
		for i := 0; i < searchBatchSize; i++ {
			idx := queryIdx % maxQueryIdx
			start := idx * dim
			end := start + dim
			batchQueries = append(batchQueries, queries[start:end])
			batchTags = append(batchTags, uint32(idx))
			queryIdx++
		}

		if len(batchQueries) > 0 {
			if err := b.rateLimiter.Wait(context.Background()); err != nil {
				fmt.Printf("Rate limit error: %v\n", err)
				continue
			}
			b.taskQueue <- Task{
				Type:      SearchTask,
				Data:      batchQueries,
				Tags:      batchTags,
				RecallAt:  config.Search.RecallAt,
				Timestamp: time.Now(),
			}
			b.searchCnt += len(batchQueries)
		}
	}
}

func (b *Bench) ConsumeTasks(numWorkers int) {

	b.index.SetQueryParams(internal.QueryParams{
		EfSearch:   uint(b.config.Search.EfSearch),
		BeamWidth:  uint(b.config.Search.BeamWidth),
		Alpha:      b.config.Search.Alpha,
		VisitLimit: uint(b.config.Search.VisitLimit),
	})

	for i := 0; i < numWorkers; i++ {
		b.wg.Add(1)
		go func(workerId int) {
			defer b.wg.Done()
			for task := range b.taskQueue {
				start := time.Now()
				switch task.Type {
				case InsertTask:
					if b.config.Workload.EnforceConsistency {
						b.rwMu.Lock()
					}
					err := b.index.BatchInsert(task.Data, task.Tags)
					if b.config.Workload.EnforceConsistency {
						b.rwMu.Unlock()
					}
					if err != nil {
						fmt.Printf("Insert error: %v\n", err)
						continue
					}
					b.insertLatencies = append(b.insertLatencies, float64(time.Since(start).Milliseconds()))
					b.insertCnt++
					b.insertPointCnt += len(task.Data)
					atomic.AddInt64(&b.globalInsertCnt, int64(len(task.Data)))
					if len(task.Tags) > 0 {
						minTag := task.Tags[0]
						maxTag := task.Tags[0]
						for _, tag := range task.Tags {
							if tag < minTag {
								minTag = tag
							}
							if tag > maxTag {
								maxTag = tag
							}
						}
						// fmt.Printf("InsertTask: tag range [%d, %d], len=%d\n", minTag, maxTag, len(task.Tags))
					}
				case SearchTask:
					if b.config.Workload.EnforceConsistency {
						b.rwMu.RLock()
					}
					results, err := b.index.BatchSearch(task.Data, uint32(task.RecallAt))
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
					b.searchLatencies = append(b.searchLatencies, float64(time.Since(start).Milliseconds()))
					b.searchCnt++
					b.searchPointCnt += len(task.Data)
				}
			}
		}(i)
	}
}

func (b *Bench) PrintProgress(totalInsert int) {
	lastPercent := -1
	for {
		current := int(atomic.LoadInt64(&b.globalInsertCnt))
		percent := current * 100 / totalInsert
		if percent != lastPercent {
			elapsed := time.Since(b.startTime).Seconds()
			insertQPS := float64(current) / elapsed
			searchQPS := float64(b.searchPointCnt) / elapsed
			fmt.Printf("Progress: %d/%d (%d%%), Insert QPS: %.2f, Search QPS: %.2f\n", current, totalInsert, percent, insertQPS, searchQPS)
			lastPercent = percent
		}
		if current >= totalInsert {
			break
		}
		time.Sleep(1 * time.Second)
	}
}

func (b *Bench) WriteResultsToCSV(elapsedSec float64, config *Config, recall float64) error {
	if err := os.MkdirAll(config.Result.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	resultPath := filepath.Join(config.Result.OutputDir, "benchmark_results.csv")

	fileExists := false
	if _, err := os.Stat(resultPath); err == nil {
		fileExists = true
	}

	file, err := os.OpenFile(resultPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open result file: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if !fileExists {
		header := []string{
			"algorithm", "threads", "batch_size", "write_ratio",
			"insert_p95_latency (ms)", "insert_p99_latency (ms)", "insert_mean_latency (ms)", "insert_qps",
			"search_p95_latency (ms)", "search_p99_latency (ms)", "search_mean_latency (ms)", "search_qps",
			"recall",
		}
		if err := writer.Write(header); err != nil {
			return fmt.Errorf("failed to write header: %v", err)
		}
	}

	var insertP95, insertP99, insertMean, insertQPS float64
	var searchP95, searchP99, searchMean, searchQPS float64

	if b.insertCnt > 0 {
		insertQPS = float64(b.insertPointCnt) / elapsedSec
		insertMean = mean(b.insertLatencies)
		insertP95 = percentile(b.insertLatencies, 0.95)
		insertP99 = percentile(b.insertLatencies, 0.99)
	}

	if b.searchCnt > 0 {
		searchQPS = float64(b.searchPointCnt) / elapsedSec
		searchMean = mean(b.searchLatencies)
		searchP95 = percentile(b.searchLatencies, 0.95)
		searchP99 = percentile(b.searchLatencies, 0.99)
	}

	row := []string{
		config.Index.IndexType,                          // algorithm
		fmt.Sprintf("%d", config.Workload.NumThreads),   // threads
		fmt.Sprintf("%d", config.Data.WriteBatchSize),   // batch_size
		fmt.Sprintf("%.2f", config.Workload.WriteRatio), // write_ratio
		fmt.Sprintf("%.2f", insertP95),                  // insert_p95_latency_ms
		fmt.Sprintf("%.2f", insertP99),                  // insert_p99_latency_ms
		fmt.Sprintf("%.2f", insertMean),                 // insert_mean_latency_ms
		fmt.Sprintf("%.2f", insertQPS),                  // insert_qps
		fmt.Sprintf("%.2f", searchP95),                  // search_p95_latency_ms
		fmt.Sprintf("%.2f", searchP99),                  // search_p99_latency_ms
		fmt.Sprintf("%.2f", searchMean),                 // search_mean_latency_ms
		fmt.Sprintf("%.2f", searchQPS),                  // search_qps
		fmt.Sprintf("%.3f", recall),                     // recall
	}

	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write data row: %v", err)
	}

	fmt.Printf("Results written to: %s\n", resultPath)
	return nil
}

func (b *Bench) CollectStats(elapsedSec float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	fmt.Println()

	if b.insertCnt > 0 {
		b.stats.InsertQPS = float64(b.insertPointCnt) / elapsedSec
		b.stats.MeanInsertLatency = mean(b.insertLatencies)
		p95 := percentile(b.insertLatencies, 0.95)
		p99 := percentile(b.insertLatencies, 0.99)
		fmt.Printf("Insert QPS: %.2f, Mean Insert Latency: %.2f ms, P95: %.2f ms, P99: %.2f ms\n", b.stats.InsertQPS, b.stats.MeanInsertLatency, p95, p99)
	} else {
		fmt.Println("No insert operations performed.")
	}
	if b.searchCnt > 0 {
		b.stats.SearchQPS = float64(b.searchPointCnt) / elapsedSec
		b.stats.MeanSearchLatency = mean(b.searchLatencies)
		p95 := percentile(b.searchLatencies, 0.95)
		p99 := percentile(b.searchLatencies, 0.99)
		fmt.Printf("Search QPS: %.2f, Mean Search Latency: %.2f ms, P95: %.2f ms, P99: %.2f ms\n", b.stats.SearchQPS, b.stats.MeanSearchLatency, p95, p99)
	} else {
		fmt.Println("No search operations performed.")
	}
}

func (b *Bench) CalcRecall(queries []float32, dataDim int, config *Config) (float64, error) {
	fmt.Println()
	if config.Result.GtPath == "" || config.Result.RecallToolPath == "" {
		fmt.Println("No ground truth or recall tool path provided, skipping recall check")
		return 0, nil
	}

	fmt.Println("Calculating recall against ground truth...")

	recallAt := config.Search.RecallAt

	outPath := config.Result.SearchResPath
	file, err := os.Create(outPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create result file: %v", err)
	}
	defer file.Close()

	numQueries := len(queries) / dataDim
	batchedQueries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		batchedQueries[i] = queries[i*dataDim : (i+1)*dataDim]
	}
	tags, err := b.index.BatchSearch(batchedQueries, uint32(recallAt))
	if err != nil {
		return 0, fmt.Errorf("batch search error: %v", err)
	}

	for i := 0; i < numQueries && i < 5; i++ {
		fmt.Printf("tags[%d]: ", i)
		for j := 0; j < len(tags[i]) && j < 5; j++ {
			fmt.Printf("%d ", tags[i][j])
		}
		fmt.Println()
	}

	n := int32(len(tags))
	k := int32(len(tags[0]))
	if err := binary.Write(file, binary.LittleEndian, n); err != nil {
		return 0, fmt.Errorf("failed to write n: %v", err)
	}
	if err := binary.Write(file, binary.LittleEndian, k); err != nil {
		return 0, fmt.Errorf("failed to write k: %v", err)
	}
	for i := 0; i < int(n); i++ {
		for j := 0; j < int(k); j++ {
			id := tags[i][j]
			if err := binary.Write(file, binary.LittleEndian, id); err != nil {
				return 0, fmt.Errorf("failed to write id: %v", err)
			}
		}
	}
	fmt.Printf("Search results written to: %s\n", outPath)

	cmd := exec.Command(config.Result.RecallToolPath,
		config.Result.GtPath,
		config.Result.SearchResPath,
		fmt.Sprintf("%d", recallAt),
	)
	output, err := cmd.CombinedOutput()
	fmt.Printf("Running calc_recall: %s\n", strings.Join(cmd.Args, " "))
	if err != nil {
		return 0, fmt.Errorf("failed to run calc_recall: %v, output: %s", err, string(output))
	}

	recall := 0.0
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "recall@") && strings.Contains(line, "%") {
			parts := strings.Split(line, "=")
			if len(parts) == 2 {
				fmt.Sscanf(parts[1], "%f%%", &recall)
			}
		}
	}
	fmt.Printf("Recall: %.4f%%\n", recall)
	return recall, nil
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
		DatasetName    string `yaml:"dataset_name"`
		MaxElements    uint64 `yaml:"max_elements"`
		BeginNum       int    `yaml:"begin_num"`
		BatchSize      int    `yaml:"batch_size"`
		WriteBatchSize int    `yaml:"write_batch_size"`
		MaxQueries     int    `yaml:"max_queries"`
		DataType       string `yaml:"data_type"`
		DataPath       string `yaml:"data_path"`
		QueryPath      string `yaml:"query_path"`
	} `yaml:"data"`

	Index struct {
		IndexType      string  `yaml:"index_type"`
		M              int     `yaml:"m"`
		EfConstruction int     `yaml:"ef_construction"`
		LevelM         float32 `yaml:"level_m"`
		Alpha          float32 `yaml:"alpha"`
		VisitLimit     int     `yaml:"visit_limit"`
		Lb             int     `yaml:"lb"`
	} `yaml:"index"`

	Search struct {
		RecallAt   uint32  `yaml:"recall_at"`
		EfSearch   uint32  `yaml:"ef_search"`
		BeamWidth  uint32  `yaml:"beam_width"`
		Alpha      float32 `yaml:"alpha"`
		VisitLimit uint32  `yaml:"visit_limit"`
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
		OutputDir      string `yaml:"output_dir"`
		GtPath         string `yaml:"gt_path"`
		SearchResPath  string `yaml:"search_res_path"`
		RecallToolPath string `yaml:"recall_tool_path"`
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
	configPath := flag.String("config", "config/config.yaml", "config file path")
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
			Dim:            dataDim,
			MaxElements:    config.Data.MaxElements,
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			Threads:        config.Workload.NumThreads,
			DataType:       internal.DataTypeFloat,
		}
		index = internal.NewIndex(internal.IndexTypeHNSW, params)
	case "parlayhnsw":
		params := internal.IndexParams{
			Dim:            dataDim,
			MaxElements:    config.Data.MaxElements,
			M:              config.Index.M,
			EfConstruction: config.Index.EfConstruction,
			LevelM:         config.Index.LevelM,
			Alpha:          config.Index.Alpha,
			Threads:        config.Workload.NumThreads,
			DataType:       internal.DataTypeFloat,
		}
		index = internal.NewIndex(internal.IndexTypeParlayHNSW, params)
	case "parlayvamana":
		params := internal.IndexParams{
			Dim:         dataDim,
			MaxElements: config.Data.MaxElements,
			M:           config.Index.M,
			Threads:     config.Workload.NumThreads,
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
		fmt.Println("Begin size:", len(preData))
		if err := index.Build(preData, preTags); err != nil {
			fmt.Printf("Warn start error: %v\n", err)
			return
		}
		fmt.Println("Index Built")
	}

	if beginNum >= int(config.Data.MaxElements) {
		fmt.Printf("BeginNum (%d) >= MaxElements (%d), no benchmark needed\n", beginNum, config.Data.MaxElements)
		return
	}

	var bench *Bench
	bench = ConcurrentBench(index, *config)
	bench.searchResults = make([]*internal.SearchResult, 0, config.Data.MaxElements)

	fmt.Printf("Threads: %d, Insert data size: %d\n", config.Workload.NumThreads, int(config.Data.MaxElements)-config.Data.BeginNum)
	fmt.Println("Start producing tasks and consuming tasks")

	start := time.Now()

	bench.startTime = time.Now()
	go func() {
		bench.ProduceTasks(data, queries, dataDim, config)
		close(bench.taskQueue)
	}()
	go bench.PrintProgress(int(config.Data.MaxElements) - config.Data.BeginNum)
	bench.ConsumeTasks(config.Workload.NumThreads)
	bench.wg.Wait()
	elapsedSec := time.Since(start).Seconds()

	fmt.Println("Streaming bench done")

	var recall float64 = 0
	if config.Result.GtPath != "" {
		recall, err = bench.CalcRecall(queries, dataDim, config)
		if err != nil {
			fmt.Printf("Failed to check recall: %v\n", err)
		}
	}

	bench.CollectStats(elapsedSec)
	if err := bench.WriteResultsToCSV(elapsedSec, config, recall); err != nil {
		fmt.Printf("Failed to write results to CSV: %v\n", err)
	}
}
