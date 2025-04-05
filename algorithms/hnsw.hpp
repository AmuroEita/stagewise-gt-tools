#include "../../hnswlib/hnswlib.h"
#include "../utils.hpp"

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class HNSW : public IndexBase<T, TagT, LabelT> {
public:
    HNSW(size_t dim, size_t max_elements, size_t M, size_t ef_construction) 
        : space(dim) {
        index = new hnswlib::HierarchicalNSW<T>(&space, max_elements, M, ef_construction);
    }

    void build(T* data, size_t num_points, std::vector<TagT>& tags) {
        for (size_t i = 0; i < num_points; i++) {
            index->addPoint(data, tags[i]);
        }
    }

    int insert_point(T* data, const TagT& tag) {
        index->addPoint(data, tag);
        return 0;
    }

    void search_with_tags(const T* query, size_t k, size_t Ls, std::vector<TagT>& result_tags) {
        auto result = index->searchKnn(query, k);
        while (!result.empty()) {
            result_tags.push_back(result.top().second);
            result.pop();
        }
    }

    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<T>* index;
};

