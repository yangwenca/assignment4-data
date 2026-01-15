import collections
import hashlib
import random
import re
import os
import unicodedata


"""
uv run pytest -k test_exact_line_deduplication
uv run pytest -k test_minhash_deduplication
"""


def hash_string_blake2(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def exact_dedup(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    myd = collections.defaultdict(int)
    for input_file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        if not os.path.isfile(input_file):
            continue
        with open(input_file, "r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                tmp = line.rstrip("\n")
                hash_line = hash_string_blake2(tmp)
                myd[hash_line] += 1

    for input_file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        if not os.path.isfile(input_file):
            continue
        with open(input_file, "r", encoding="utf-8", errors="replace") as fin, \
            open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                tmp = line.rstrip("\n")
                hash_line = hash_string_blake2(tmp)
                if myd[hash_line] == 1:
                    fout.write(line)


# Text normalization
def normalize_text(text: str) -> str:
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove accents (diacritics)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )
    # Unicode NFD normalization
    text = unicodedata.normalize("NFD", text)
    return text


def word_ngrams(text: str, n: int):
    words = text.split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


# MinHash
# minhash(hi,S) = min(hi(s1),hi(s2),...,hi(sn))
# s1, s2,...,sn are n-grams for S
def minhash_signature(ngrams: set, num_hashes: int) -> list[int]:
    signature = []
    for i in range(num_hashes):
        min_val = float('inf')
        for ng in ngrams:
            h = hashlib.md5((str(i) + ng).encode("utf-8")).hexdigest()
            val = int(h, 16)
            min_val = min(val, min_val)
        signature.append(min_val)
    return signature


def fuzzy_deduplicate(
    input_paths: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngram_size: int,
    jaccard_threshold: float,
    output_dir: os.PathLike,
):
    assert num_hashes % num_bands == 0
    rows_per_band = num_hashes // num_bands
    seed = 42
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    buckets = collections.defaultdict(list)
    total = []

    for idx, path in enumerate(input_paths):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            docs = [line.rstrip("\n") for line in f]

        normalized = [normalize_text(d) for d in docs]
        normalized = ' '.join(normalized)
        ngram_sets = word_ngrams(normalized, ngram_size)
        # compute minhash for each document
        signatures = minhash_signature(ngram_sets, num_hashes)
        total.append(signatures)
        for b in range(num_bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_sig = tuple(signatures[start:end])
            bucket_id = (b, band_sig)
            buckets[bucket_id].append(idx)

    parent = list(range(len(input_paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Candidate comparison
    for bucket_docs in buckets.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                d1, d2 = bucket_docs[i], bucket_docs[j]
                count = sum([1 for ta, tb in zip(total[d1], total[d2]) if ta == tb])
                jaccard = count / len(total[d1])
                if jaccard > jaccard_threshold:
                    union(d1, d2)

    clusters = collections.defaultdict(list)
    for i in range(len(input_paths)):
        clusters[find(i)].append(i)

    retained = set()
    for cluster_docs in clusters.values():
        retained.add(random.choice(cluster_docs))

    for idx in retained:
        input_path = input_paths[idx]
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
                fout.writelines(fin)
