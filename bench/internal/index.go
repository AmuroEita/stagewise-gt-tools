package internal

// #cgo CXXFLAGS: -I${SRCDIR}/../algorithms
// #cgo LDFLAGS: -L${SRCDIR}/../build/lib -lindex
// #cgo CXXFLAGS: -std=c++14
// #include <stdlib.h>
// #include "../algorithms/index_cgo.hpp"
import "C"
import (
	"fmt"
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
	DataType    DataType
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
	}
	return &Index{
		ptr: C.create_index(C.IndexType(indexType), cParams),
	}
}

func (i *Index) Close() {
	if i.ptr != nil {
		C.destroy_index()
		i.ptr = nil
	}
}

func (i *Index) Build(data [][]float32, tags []uint32) error {
	if len(data) == 0 || len(tags) == 0 {
		return nil
	}
	numPoints := len(data)
	dim := len(data[0])
	flatData := make([]float32, numPoints*dim)
	for j, vec := range data {
		copy(flatData[j*dim:], vec)
	}
	result := C.build_index(
		(*C.float)(&flatData[0]),
		C.size_t(len(tags)),
		(*C.uint32_t)(&tags[0]),
	)
	if result != 0 {
		return fmt.Errorf("build index failed with code: %d", result)
	}
	return nil
}

func (i *Index) InsertPoint(point []float32, tag uint32) error {
	if len(point) == 0 {
		return nil
	}

	result := C.insert_point(
		(*C.float)(&point[0]),
		C.uint32_t(tag),
	)

	if result != 0 {
		return fmt.Errorf("insert point failed with code: %d", result)
	}
	return nil
}

func (i *Index) SetQueryParams(Ls uint) {
	C.set_query_params(C.size_t(Ls))
}

func (i *Index) Search(query []float32, k, Ls uint) ([]uint32, error) {
	if len(query) == 0 {
		return nil, nil
	}

	results := make([]uint32, k)
	result := C.search_with_tags(
		(*C.float)(&query[0]),
		C.size_t(k),
		C.size_t(Ls),
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

	queryPtrsC := C.malloc(C.size_t(len(queries)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(unsafe.Pointer(queryPtrsC))
	queryPtrsGo := (*[1 << 30]*C.float)(queryPtrsC)[:len(queries):len(queries)]
	for j, q := range queries {
		queryPtrsGo[j] = (*C.float)(&q[0])
	}

	results := make([][]uint32, len(queries))
	for j := range results {
		results[j] = make([]uint32, k)
	}

	resultPtrsC := C.malloc(C.size_t(len(results)) * C.size_t(unsafe.Sizeof(uintptr(0))))
	defer C.free(unsafe.Pointer(resultPtrsC))
	resultPtrsGo := (*[1 << 30]*C.uint32_t)(resultPtrsC)[:len(results):len(results)]
	for j, r := range results {
		resultPtrsGo[j] = (*C.uint32_t)(&r[0])
	}

	result := C.batch_search(
		(**C.float)(queryPtrsC),
		C.size_t(len(queries)),
		C.uint32_t(k),
		C.uint32_t(Ls),
		(**C.uint32_t)(resultPtrsC),
	)

	if result != 0 {
		return nil, fmt.Errorf("batch search failed with code: %d", result)
	}
	return results, nil
}
