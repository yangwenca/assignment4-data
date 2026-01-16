import concurrent.futures
import multiprocessing
import os
import re
from tqdm import tqdm

import numpy as np
from cs336_data.training import get_text_from_wet
from transformers import AutoTokenizer

"""
filter_data
(b)
700 seconds per WET files, 12 cpu in parallel
5000 WET files take 291660 seconds, 81 hours
100,000 WET files take 1620 hours

inspect filtered data
(a)
some of them have low quality, might be too short, or too random
some of them are duplicates from each other

(b)
languages are not english, some are too short or too long

(c)
need to add deduplicateds in the model

tokenize_data
1 wet file 100k tokens
5000 WET, should give 500M tokens in total
"""


def process_single_wet_file(wet_path: str, output_dir: str, lang="en") -> str:
    assert wet_path.endswith(".wet.gz")
    ans = get_text_from_wet(wet_path, 2000)
    output_name = wet_path.split('/')[-1].replace(".gz", "") + ".txt"
    output_path = output_dir + output_name
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in ans:
            if len(text.encode("utf-8")) >= 1020:
                continue
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = re.sub(r'\s+', ' ', text.strip())
            f.write(text + "\n")
    return output_path


def filter_wet_directory(output_dir: str) -> None:
    wet_filepaths = [
        "/Users/YangWen/Documents/Code/github/data/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz",
        "/Users/YangWen/Documents/Code/github/data/data/CC/CC-MAIN-20250417135010-20250417165010-00065_copy.warc.wet.gz",
    ]
    os.makedirs(output_dir, exist_ok=True)
    num_cpus = os.cpu_count()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    futures = []
    for wet_filepath in wet_filepaths:
        # For each warc.wet.gz filepath, submit a job to the executor and get a future back
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            output_dir,
        )
        # Store the futures
        futures.append(future)

    # Iterate over the completed futures as they finish, using a progress bar # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
    ):
        output_file = future.result()
        print(f"Output file written: {output_file}")


def tokenize_line_and_add_eos(line):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer.encode(line) + [tokenizer.eos_token_id]


def tokenize(input_path: str, output_path: str):
    with open(input_path) as f:
        lines = f.readlines()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 100
    results = []
    for result in tqdm(
        pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
        total=len(lines),
        desc="Tokenizing lines",
    ):
        results.append(result)

    pool.close()
    pool.join()

    # Flatten the list of ids and convert to numpy array
    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)


if __name__ == "__main__":
    filter_wet_directory(
        output_dir="/Users/YangWen/Documents/Code/github/data/data/output/",
    )
    tokenize(
        input_path="/Users/YangWen/Documents/Code/github/data/data/output/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.txt",
        output_path="/Users/YangWen/Documents/Code/github/data/data/output/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.npy"
    )
