package main

import (
    "fmt"
    "sync"
    "time"
)

type TaskType int

const (
    InsertTask TaskType =(pr * 0)
    SearchTask
)

type Task struct {
    Type      TaskType
    Data      []float32
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
    BatchSearch(queries [][]float32, recallAt, Ls uint32) ([][]uint32, error)
}

type Bench struct {
    taskQueue       chan Task
    index           Index
    stats           Stat
    mu              sync.Mutex
    rwMu            sync.RWMutex // Added: read-write lock for index access
    wg              sync.WaitGroup
    insertCnt       int
    searchCnt       int
    insertLatencies []float64
    searchLatencies []float64
}

func NewBench(index Index, queueSize int) *Bench {
    return &Bench{
        taskQueue:       make(chan Task, queueSize),
        index:           index,
        insertLatencies: make([]float64, 0),
        searchLatencies: make([]float64, 0),
    }
}

func (b *Bench) ProduceTasks(data [][]float32, queries [][]float32, beginNum int, writeRatio float64, batchSize int, recallAt, Ls uint32) {
    defer close(b.taskQueue)

    insertTotal := len(data) - beginNum
    searchTotal := int(float64(insertTotal) * (1-writeRatio) / writeRatio)
    searchBatchSize := int(float64(batchSize) * (1-writeRatio) / writeRatio)

    startInsertOffset := 0
    startSearchOffset := 0
    queryIdx := uint32(0)

    for startInsertOffset < insertTotal || startSearchOffset < searchTotal {
        endInsertOffset := min(startInsertOffset+batchSize, insertTotal)
        if startInsertOffset < endInsertOffset {
            batchData := make([][]float32, 0, batchSize)
            batchTags := make([]uint32, 0, batchSize)
            for i := startInsertOffset; i < endInsertOffset; i++ {
                batchData = append(batchData, data[beginNum+i])
                batchTags = append(batchTags, uint32(beginNum+i))
            }
            b.taskQueue <- Task{
                Type:      InsertTask,
                Data:      batchData,
                Tags:      batchTags,
                Timestamp: time.Now(),
            }
            startInsertOffset = endInsertOffset
            b.insertCnt += len(batchData)
        }

        endSearchOffset := min(startSearchOffset+searchBatchSize, searchTotal)
        if startSearchOffset < endSearchOffset {
            batchQueries := make([][]float32, 0, searchBatchSize)
            for i := startSearchOffset; i < endSearchOffset; i++ {
                queryIdx = (queryIdx + 1) % uint32(len(queries))
                batchQueries = append(batchQueries, queries[queryIdx])
            }
            b.taskQueue <- Task{
                Type:      SearchTask,
                Data:      batchQueries,
                RecallAt:  recallAt,
                Ls:        Ls,
                Timestamp: time.Now(),
            }
            startSearchOffset = endSearchOffset
            b.searchCnt += len(batchQueries)
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
                    b.rwMu.Lock() // Exclusive write lock
                    err := b.index.BatchInsert(task.Data, task.Tags)
                    b.rwMu.Unlock()
                    if err != nil {
                        fmt.Printf("Insert error: %v\n", err)
                        continue
                    }
                    b.mu.Lock()
                    b.insertLatencies = append(b.insertLatencies, float64(time.Since(start).Microseconds()))
                    b.mu.Unlock()
                case SearchTask:
                    b.rwMu.RLock() // Shared read lock
                    results, err := b.index.BatchSearch(task.Data, task.RecallAt, task.Ls)
                    b.rwMu.RUnlock()
                    if err != nil {
                        fmt.Printf("Search error: %v\n", err)
                        continue
                    }
                    b.mu.Lock()
                    b.searchLatencies = append(b.searchLatencies, float64(time.Since(start).Microseconds()))
                    b.mu.Unlock()
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

func main() {
    var index Index
    bench := NewBench(index, 1000)

    data := make([][]float32, 10000)
    queries := make([][]float32, 1000)
    beginNum := 5000
    writeRatio := 0.5
    batchSize := 100
    recallAt := uint32(10)
    Ls := uint32(50)
    numThreads := 16

    start := time.Now()

    go bench.ProduceTasks(data, queries, beginNum, writeRatio, batchSize, recallAt, Ls)
    bench.ConsumeTasks(numThreads)

    bench.wg.Wait()
    elapsedSec := time.Since(start).Seconds()

    bench.CollectStats(elapsedSec)
}
