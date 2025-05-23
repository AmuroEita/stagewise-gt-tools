package internal

import (
	"encoding/binary"
	"fmt"
	"os"
)

type SearchResult struct {
	InsertOffset uint64
	QueryIdx     uint64
	Tags         []uint32
	Distances    []float32
}

func PreAllocateSearchResults(dataNum uint64, writeRatio float64) []*SearchResult {
	capacity := int(float64(dataNum) * (1.0/writeRatio - 1.0))
	return make([]*SearchResult, 0, capacity)
}

func NewSearchResult(offset, idx uint64, tags []uint32) *SearchResult {
	return &SearchResult{
		InsertOffset: offset,
		QueryIdx:     idx,
		Tags:         tags,
		Distances:    make([]float32, 0),
	}
}

func NewSearchResultWithDistances(offset, idx uint64, tags []uint32, distances []float32) *SearchResult {
	return &SearchResult{
		InsertOffset: offset,
		QueryIdx:     idx,
		Tags:         tags,
		Distances:    distances,
	}
}

func GetBinMetadata(filename string, numPoints *uint64, dimensions *uint64) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var metadata [2]uint32
	if err := binary.Read(file, binary.LittleEndian, &metadata); err != nil {
		return fmt.Errorf("failed to read metadata: %v", err)
	}

	*numPoints = uint64(metadata[0])
	*dimensions = uint64(metadata[1])

	fmt.Printf("File %s contains %d points and %d dimensions\n",
		filename, *numPoints, *dimensions)

	return nil
}

func LoadAlignedBin(binFile string) (data []float32, npts, dim, roundedDim uint32, err error) {
	fmt.Printf("Reading bin file %s ...", binFile)

	file, err := os.Open(binFile)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("failed to get file info: %v", err)
	}
	actualFileSize := fileInfo.Size()

	var nptsI32, dimI32 int32
	if err := binary.Read(file, binary.LittleEndian, &nptsI32); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("failed to read npts: %v", err)
	}
	if err := binary.Read(file, binary.LittleEndian, &dimI32); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("failed to read dim: %v", err)
	}

	npts = uint32(nptsI32)
	dim = uint32(dimI32)

	expectedFileSize := int64(npts*dim*4 + 8)
	if actualFileSize != expectedFileSize {
		return nil, 0, 0, 0, fmt.Errorf("file size mismatch: actual=%d, expected=%d (npts=%d, dim=%d)",
			actualFileSize, expectedFileSize, npts, dim)
	}

	roundedDim = (dim + 7) & ^uint32(7)
	fmt.Printf("Metadata: #pts = %d, #dims = %d, aligned_dim = %d... ", npts, dim, roundedDim)

	allocSize := npts * roundedDim * 4
	fmt.Printf("Allocating memory of %d bytes... ", allocSize)
	data = make([]float32, npts*roundedDim)
	fmt.Print("done. Copying data...")

	for i := uint32(0); i < npts; i++ {
		row := data[i*roundedDim : i*roundedDim+dim]
		if err := binary.Read(file, binary.LittleEndian, row); err != nil {
			return nil, 0, 0, 0, fmt.Errorf("failed to read data point %d: %v", i, err)
		}
		for j := dim; j < roundedDim; j++ {
			data[i*roundedDim+j] = 0
		}
	}
	fmt.Println(" done.")

	return data, npts, dim, roundedDim, nil
}
