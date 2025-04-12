#include <cstdint>

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
class IndexBase {
   public:
    virtual ~IndexBase() = default;
    virtual void build(T *data, size_t num_points, std::vector<TagT> &tags) = 0;
    virtual int insert_point(T *point, const TagT &tag) = 0;
    virtual void set_query_params(const size_t Ls) = 0;
    virtual void search_with_tags(const T *query, size_t k, size_t Ls,
                                  std::vector<TagT> &res_tags) = 0;
};