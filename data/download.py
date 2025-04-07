import os
import tarfile
import wget

urls = [
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/gist.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/sift10m.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/msong.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/glove1.2m.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/glove2.2m.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/imagenet.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/netflix.tar.gz",
    "https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/word2vec.tar.gz",
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
]

download_dir = "downloads"
os.makedirs(download_dir, exist_ok=True)

for url in urls:
    try:
        filename = os.path.join(download_dir, url.split('/')[-1])
        
        print(f"Downloading {url}...")
        wget.download(url, out=filename)
        print(f"\nDownloaded {filename}")
        
        print(f"Extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print(f"Extracted {filename}")
        
        os.remove(filename)
        print(f"Removed {filename}")
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

print("All downloads and extractions completed!")