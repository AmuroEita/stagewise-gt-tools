paths = {
    ("deep1M/deep1M_base.fvecs", "deep1M/deep1M_query.fvecs", "deep1M/deep1M_query_1k.fvecs", "deep1M/deep1M_base_100_1k.gt"),
    ("gist/gistbase.fvecs", "gist/gist_query.fvecs", "gist/gist_query_1k.fvecs", "gist/gist_base_100_1k.gt"),
    ("glove1.2m/glove1.2m_base.fvecs", "glove1.2m/glove1.2m_query.fvecs", "glove1.2m/glove1.2m_query_1k.fvecs", "glove1.2m/glove1.2m_base_100_1k.gt"),
    ("glove2.2m/glove2.2m_base.fvecs", "glove2.2m/glove2.2m_query.fvecs", "glove2.2m/glove2.2m_query_1k.fvecs", "glove2.2m/glove2.2m_base_100_1k.gt"),
    ("msong/msong_base.fvecs", "msong/msong_query.fvecs", "msong/msong_query_1k.fvecs", "msong/msong_base_100_1k.gt"),
    ("netflix/netflix_base.fvecs", "netflix/netflix_query.fvecs", "netflix/netflix_query_1k.fvecs", "netflix/netflix_base_100_1k.gt"),
    ("sift10m/sift10m_base.fvecs", "sift10m/sift10m_query.fvecs", "sift10m/sift10m_query_1k.fvecs", "sift10m/sift10m_base_100_1k.gt"),
    ("word2vec/word2vec_base.fvecs", "word2vec/word2vec_query.fvecs", "word2vec/word2vec_query_1k.fvecs", "word2vec/word2vec_base_100_1k.gt"),
    ("sift/sift_base.fvecs", "sift/sift_query.fvecs", "sift/sift_query_1k.fvecs", "sift/sift_base_100_1k.gt"),
}

for a, b, c, d in paths:
    cmd = f"./crop {b} 1000 {c}"
    print(f"Executing: {cmd}")
    import os
    os.system(cmd)

for a, b, c, d in paths:
    cmd = f"./compute_gt --base_path {a} --query_path {c} --gt_path {d} --data_type float --k 10"
    print(f"Executing: {cmd}")
    import os
    os.system(cmd)