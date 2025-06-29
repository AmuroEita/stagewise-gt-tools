package internal

// #cgo CXXFLAGS: -I${SRCDIR}/../algorithms -std=c++17
// #cgo LDFLAGS: -L${SRCDIR}/../build/lib -lindex
// #include <stdlib.h>
// #include "../algorithms/index_cgo.hpp"
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeParlayHNSW
	IndexTypeParlayVamana
	IndexTypeCCHNSW
)

type DataType int

const (
	DataTypeFloat DataType = iota
	DataTypeInt8
	DataTypeUint8
)

type IndexParams struct {
	Dim         int
	MaxElements uint64
	M           int
	Lb          int
	Threads     int
	DataType    DataType
}

type QueryParams struct {
	Ls         uint
	BeamWidth  uint
	Alpha      float32
	VisitLimit uint
}

type Index struct {
	ptr unsafe.Pointer
}

func NewIndex(indexType IndexType, params IndexParams) *Index {
	cParams := C.IndexParams{
		dim:          C.size_t(params.Dim),
		max_elements: C.size_t(params.MaxElements),
		M:            C.size_t(params.M),
		Lb:           C.size_t(params.Lb),
		num_threads:  C.size_t(params.Threads),
	}
	return &Index{
		ptr: C.create_index(C.IndexType(indexType), cParams),
	}
}

func (i *Index) Close() {
	if i.ptr != nil {
		C.destroy_index(i.ptr)
		i.ptr = nil
	}
}

func (i *Index) Build(data [][]float32, tags []uint32) error {
	if len(data) == 0 || len(tags) == 0 {
		return nil
	}

	startTime := time.Now()

	numPoints := len(data)
	dim := len(data[0])
	flatData := make([]float32, numPoints*dim)
	for j, vec := range data {
		copy(flatData[j*dim:], vec)
	}
	result := C.build(
		i.ptr,
		(*C.float)(&flatData[0]),
		(*C.uint32_t)(&tags[0]),
		C.size_t(len(tags)),
	)

	buildTime := time.Since(startTime)
	qps := float64(numPoints) / buildTime.Seconds()

	fmt.Printf("Build completed: %d points in %v (%.2f QPS)\n", numPoints, buildTime, qps)

	if result != 0 {
		return fmt.Errorf("build index failed with code: %d", result)
	}
	return nil
}

func (i *Index) Insert(point []float32, tag uint32) error {
	if len(point) == 0 {
		return nil
	}

	result := C.insert(
		i.ptr,
		(*C.float)(&point[0]),
		C.uint32_t(tag),
	)

	if result != 0 {
		return fmt.Errorf("insert point failed with code: %d", result)
	}
	return nil
}

func (i *Index) SetQueryParams(params QueryParams) {
	cParams := C.C_QParams{
		Ls:          C.size_t(params.Ls),
		beam_width:  C.size_t(params.BeamWidth),
		alpha:       C.float(params.Alpha),
		visit_limit: C.size_t(params.VisitLimit),
	}
	C.set_query_params(i.ptr, cParams)
}

func (i *Index) Search(query []float32, k uint, params QueryParams) ([]uint32, error) {
	results := make([]uint32, k)
	cParams := C.C_QParams{
		Ls:          C.size_t(params.Ls),
		beam_width:  C.size_t(params.BeamWidth),
		alpha:       C.float(params.Alpha),
		visit_limit: C.size_t(params.VisitLimit),
	}
	result := C.search(
		i.ptr,
		(*C.float)(&query[0]),
		C.size_t(k),
		cParams,
		(*C.uint32_t)(&results[0]),
	)
	if result != 0 {
		return nil, fmt.Errorf("search failed with code: %d", result)
	}
	return results, nil
}

func (i *Index) BatchInsert(batchData [][]float32, batchTags []uint32) error {
	if len(batchData) == 0 || len(batchTags) == 0 {
		return nil
	}

	numPoints := len(batchData)
	if numPoints == 0 {
		return nil
	}
	dim := len(batchData[0])
	flatData := make([]float32, numPoints*dim)
	for j, vec := range batchData {
		copy(flatData[j*dim:], vec)
	}

	result := C.batch_insert(
		i.ptr,
		(*C.float)(&flatData[0]),
		(*C.uint32_t)(&batchTags[0]),
		C.size_t(numPoints),
	)

	if result != 0 {
		return fmt.Errorf("batch insert failed with code: %d", result)
	}
	return nil
}

func (i *Index) BatchSearch(queries [][]float32, k, Ls uint) ([][]uint32, error) {
	if len(queries) == 0 {
		return nil, nil
	}

	dim := len(queries[0])
	flatQueries := make([]float32, 0, len(queries)*dim)
	for _, q := range queries {
		flatQueries = append(flatQueries, q...)
	}

	results := make([][]uint32, len(queries))
	for j := range results {
		results[j] = make([]uint32, k)
	}

	resultPtrs := make([]*C.uint32_t, len(queries))
	for j := range results {
		resultPtrs[j] = (*C.uint32_t)(&results[j][0])
	}
	resultPtrsC := C.malloc(C.size_t(len(queries)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(resultPtrsC)
	ptrSlice := (*[1 << 30]*C.uint32_t)(resultPtrsC)[:len(queries):len(queries)]
	for j, p := range resultPtrs {
		ptrSlice[j] = p
	}

	cParams := C.C_QParams{
		Ls:          C.size_t(Ls),
		beam_width:  0,
		alpha:       0,
		visit_limit: 0,
	}

	result := C.batch_search(
		i.ptr,
		(*C.float)(&flatQueries[0]),
		C.uint32_t(k),
		cParams,
		C.size_t(len(queries)),
		(**C.uint32_t)(resultPtrsC),
	)

	if result != 0 {
		return nil, fmt.Errorf("batch search failed with code: %d", result)
	}
	return results, nil
}
