package internal

// #cgo CXXFLAGS: -I${SRCDIR}/../algorithms
// #cgo LDFLAGS: -L${SRCDIR}/../algorithms -lindex
// #cgo CXXFLAGS: -std=c++14
// #include "../algorithms/index_cgo.hpp"
import "C"
import (
	"fmt"
	"unsafe"
)

type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeVAMANA
	IndexTypePARLAYANN
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
		C.destroy_index(i.ptr)
		i.ptr = nil
	}
}

func (i *Index) Build(data []float32, tags []uint32) error {
	if len(data) == 0 || len(tags) == 0 {
		return nil
	}

	result := C.build_index(
		i.ptr,
		(*C.float)(&data[0]),
		C.size_t(len(data)),
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
		i.ptr,
		(*C.float)(&point[0]),
		C.uint32_t(tag),
	)

	if result != 0 {
		return fmt.Errorf("insert point failed with code: %d", result)
	}
	return nil
}

func (i *Index) SetQueryParams(Ls uint) {
	C.set_query_params(i.ptr, C.size_t(Ls))
}

func (i *Index) Search(query []float32, k, Ls uint) ([]uint32, error) {
	if len(query) == 0 {
		return nil, nil
	}

	results := make([]uint32, k)
	result := C.search_with_tags(
		i.ptr,
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

	// 创建指向每个向量的指针数组
	dataPtrs := make([]*float32, len(batchData))
	for j := range batchData {
		dataPtrs[j] = &batchData[j][0]
	}

	result := C.batch_insert(
		i.ptr,
		(**C.float)(unsafe.Pointer(&dataPtrs[0])),
		(*C.uint32_t)(&batchTags[0]),
		C.size_t(len(batchData)),
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

	// 创建指向每个查询向量的指针数组
	queryPtrs := make([]*float32, len(queries))
	for j := range queries {
		queryPtrs[j] = &queries[j][0]
	}

	// 创建结果数组
	results := make([][]uint32, len(queries))
	for j := range results {
		results[j] = make([]uint32, k)
	}

	result := C.batch_search(
		i.ptr,
		(**C.float)(unsafe.Pointer(&queryPtrs[0])),
		C.size_t(len(queries)),
		C.uint32_t(k),
		C.uint32_t(Ls),
		(**C.uint32_t)(unsafe.Pointer(&results[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("batch search failed with code: %d", result)
	}
	return results, nil
}
