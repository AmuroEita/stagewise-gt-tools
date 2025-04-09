paths = {
    ("deep1M/deep1M_base.fvecs", "deep1M/deep1M_query.fvecs", "deep1M/deep1M_query_1k.fvecs", "deep1M/deep1M_base_1k_b100.gt20", "deep1M/deep1M_base_1k.gt20"),
    ("gist/gistbase.fvecs", "gist/gist_query.fvecs", "gist/gist_query_1k.fvecs", "gist/gist_base_1k_b100.gt20", "gist/gist_base_1k.gt20"),
    ("glove1.2m/glove1.2m_base.fvecs", "glove1.2m/glove1.2m_query.fvecs", "glove1.2m/glove1.2m_query_1k.fvecs", "glove1.2m/glove1.2m_base_1k_b100.gt20", "glove1.2m/glove1.2m_base_1k.gt20"),
    ("glove2.2m/glove2.2m_base.fvecs", "glove2.2m/glove2.2m_query.fvecs", "glove2.2m/glove2.2m_query_1k.fvecs", "glove2.2m/glove2.2m_base_1k_b100.gt20", "glove2.2m/glove2.2m_base_1k.gt20"),
    ("msong/msong_base.fvecs", "msong/msong_query.fvecs", "msong/msong_query_1k.fvecs", "msong/msong_base_1k_b100.gt20", "msong/msong_base_1k.gt20"),
    ("netflix/netflix_base.fvecs", "netflix/netflix_query.fvecs", "netflix/netflix_query_1k.fvecs", "netflix/netflix_base_1k_b100.gt20", "netflix/netflix_base_1k.gt20"),
    ("word2vec/word2vec_base.fvecs", "word2vec/word2vec_query.fvecs", "word2vec/word2vec_query_1k.fvecs", "word2vec/word2vec_base_1k_b100.gt20", "word2vec/word2vec_base_1k.gt20"),
    ("sift/sift_base.fvecs", "sift/sift_query.fvecs", "sift/sift_query_1k.fvecs", "sift/sift_base_1k_b100.gt20" "sift/sift_base_1k.gt20"),
}

for a, b, c, d, e in paths:
    cmd = f"../build/crop {b} 1000 {c}"
    print(f"Executing: {cmd}")
    import os
    os.system(cmd)

for a, b, c, d, e in paths:
    cmd = f"../build/compute_gt --base_path {a} --query_path {c} --batch_gt_path {d} --gt_path {e}--data_type float --k 20"
    print(f"Executing: {cmd}")
    import os
    os.system(cmd)
