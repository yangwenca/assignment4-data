"""
Train a fastText quality classifier
wiki: Wiki pages as positive examples
cc: random pages from Common Crawl
"""
from typing import Any
import gzip
import os
import re
import subprocess
import tempfile

import fasttext
from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text, get_language, has_alpha

"""
sample file format

WARC/1.0
WARC-Type: conversion
WARC-Target-URI: http://10www.chinatikfans.com/home.php?mod=space&uid=4693&do=blog&classid=104&view=me
WARC-Date: 2025-04-17T14:32:36Z
WARC-Record-ID: <urn:uuid:530c3836-455b-4443-8c32-a4f91dca9e88>
WARC-Refers-To: <urn:uuid:8673e068-bed0-45ac-a821-b6cd217f8a1d>
WARC-Block-Digest: sha1:NHSN24WNSZZQUEVZZVYJA7Q24C66YZPQ
WARC-Identified-Content-Language: zho,eng
Content-Type: text/plain
Content-Length: 3729

lily_zl的日志 - &#1769;杰西达邦中国影迷会&#1769; - Powered by Discuz!
设为首页收藏本站
开启辅助访问 切换到窄版
帐号 自动登录 找回密码
"""

probability = 0.7
def get_text_from_wet(wet_path: str, num_samples: int) -> list[str]:
    texts = []
    with gzip.open(wet_path, 'rt', encoding='utf-8', errors='ignore') as f:
        current_txt = []
        in_content = False
        for line in f:
            if len(texts) >= num_samples:
                break
            line = line.strip()
            if line.startswith('WARC'):
                if in_content and current_txt:
                    content = '\n'.join(current_txt).strip()
                    content = re.sub(r'\s+', ' ', content)
                    content = ' '.join(content.split()[:1000])
                    lang, prob = get_language(content)
                    if lang == 'en' and prob > probability:
                        texts.append(content)
                current_txt = []
                in_content = False
            elif line.startswith('Content-Length:'):
                in_content = True
            elif in_content:
                current_txt.append(line)
    return texts


def get_text_from_warc(warc_path: str, num_samples: int) -> list[str]:
    ans = []
    stream = GZipStream(FileStream(warc_path, 'rb'))
    for record in ArchiveIterator(stream):
        if record.record_type == WarcRecordType.response:
            payload_bytes = record.reader.read()
            text = extract_text(payload_bytes)
            if text is None:
                continue
            text = re.sub(r'\s+', ' ', text.strip())
            lang, prob = get_language(text)
            if prob < probability:
                continue
            if lang != "en":
                continue
            # filter out based on percentage of alpha words
            text_tmp = text.split()
            length = len(text_tmp)
            if length < 500:
                continue
            alpha_words = sum(has_alpha(word) for word in text_tmp)
            if alpha_words / length > 0.6:
                ans.append(text)
            if len(ans) >= num_samples:
                break
    return ans


def get_text_from_wiki(wiki_path: str, num_samples: int) -> list[str]:
    urls = []
    # get urls from wiki
    with gzip.open(wiki_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(urls) >= num_samples:
                break
            url = line.strip()
            if url:
                urls.append(url)
    # very easy to timeout, get 50 urls as a group
    per_group = 50
    groups = len(urls) // per_group
    texts = []
    for idx in range(groups):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for url in urls[idx*per_group:(idx+1)*per_group]:
                f.write(url + '\n')
            urls_file_name = f.name

        with tempfile.TemporaryDirectory() as temp_dir:
            warc_file_name = os.path.join(temp_dir, 'wiki.file')
            cmd = [
                'wget',
                '--timeout=5',
                '-i',
                urls_file_name,
                f'--warc-file={warc_file_name}',
                '-O',
                '/dev/null',
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                warc_gz_file_name = warc_file_name + '.warc.gz'
                texts += get_text_from_warc(warc_gz_file_name, num_samples)
                if len(texts) >= num_samples:
                    break
            except Exception as e:
                print(f"group number {idx}, wget command failed with error {e}")
    return texts


def get_data(wet_path: str, wiki_path: str, output_path: str, num_samples: int) -> None:
    cc_file = output_path + "_cc"
    cc_samples = get_text_from_wet(wet_path=wet_path, num_samples=num_samples)
    print(f"number of cc samples are {len(cc_samples)}")
    with open(cc_file, 'w', encoding='utf-8') as f:
        for text in cc_samples:
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = re.sub(r'\s+', ' ', text.strip())
            f.write(f"__label__cc {text}" + "\n")

    wiki_file = output_path + "_wiki"
    wiki_samples = get_text_from_wiki(wiki_path=wiki_path, num_samples=num_samples * 3)
    print(f"number of wiki samples are {len(wiki_samples)}")
    with open(wiki_file, 'w', encoding='utf-8') as f:
        for text in wiki_samples:
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = re.sub(r'\s+', ' ', text.strip())
            f.write(f"__label__wiki {text}" + "\n")

    with open(cc_file, "r", encoding="utf-8") as f1, \
        open(wiki_file, "r", encoding="utf-8") as f2, \
        open(output_path , "w", encoding="utf-8") as out:

        out.writelines(f1)
        out.writelines(f2)


def training_model(training_file: str, model_file: str) -> None:
    model = fasttext.train_supervised(
        input=training_file,
        lr=0.1,
        epoch=50,
        wordNgrams=2,
    )
    model.save_model(model_file)


def predict_quality(text: str) -> tuple[Any, float]:
    tmp = text.split()
    text = " ".join(tmp)
    model = fasttext.load_model('/Users/YangWen/Documents/Code/github/data/data/CC/classifier_own.bin')
    labels, probabilities = model.predict(text)
    label = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    return (label, confidence)


def main():
    training_path='/Users/YangWen/Documents/Code/github/data/data/CC/classifier_data.txt'
    get_data(
        wet_path='/Users/YangWen/Documents/Code/github/data/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz',
        wiki_path='/Users/YangWen/Documents/Code/github/data/data/CC/enwiki-20240420-extracted_urls.txt.gz',
        output_path=training_path,
        num_samples=50,
    )

    model_path = '/Users/YangWen/Documents/Code/github/data/data/CC/classifier_own.bin'
    training_model(training_file=training_path, model_file=model_path)


if __name__ == "__main__":
    main()
