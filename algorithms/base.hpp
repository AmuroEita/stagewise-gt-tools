#pragma once

#include <cstdint>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
public:
    virtual ~IndexBase() = default;
    void build(T* data, size_t num_points, const std::vector<TagT>& tags) {

    }

    int insert_point(T* point, TagT tag) { 
        return 0; 
    }

    void search_with_tags(const T* query, size_t k, size_t Ls, TagT* tags, std::vector<T*>& res) {
        
    }
};