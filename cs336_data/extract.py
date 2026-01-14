"""
look at common crawl

(a)
URL is http://0371rykj.com/ipfhsb/34.html
No, it is no longer accessible.
a company's website, it sells some kinds of experiments equipments

(b)
harmful information
the data quality is too low, the model trained by this data might have poor performance.
this WET file is much cleaner than WARC file, the model can get product information.

(c)
Application domain: when users are asking questions about similar products.
Not relevant: sports

(d)
it takes me at least 30 documents until I see a high quality example

language: zho
domain: company products
type of page: text/plain
miscellaneous: 

language: zho, eng
domain: movie
type of page: text/plain
miscellaneous

language: eng
domain: academia
type of page: text/plain
miscellaneous

language: zho
domain: gambling
type of page: text/plain
miscellaneous

language: jpn, zho
domain: company products
type of page: text/plain
miscellaneous: 
"""

from typing import Any

from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext


def extract_text(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    bytes_str = html_bytes.decode(encoding)
    return extract_plain_text(bytes_str)


"""
extract_text
(b)
My text extraction function can extract the text, but it containts lots of empty lines.
It does not split the lines as good as WET files.
WET file does a better job.
"""


"""
language identification
(b)
False positives (wrong language accepted)
non target language text is mistakenly labeled as the target language.
language contamination
reduced fluency
code switching artifacts
bias amplification

Mixed language / code switching errors
documents containing multiple languages are forced into a single label
loss of multilingual capability, confusing responses

mitigation stratgies
use ensembles, not a single language model,
combine multiple signals

confidence thresholding
require high confidence for hard filtering
route low confidence cases to a secondary classifier, a mixed/unknown bucket
(c)
20 random files
no classifier errors, 10% of documents is english
suitable claissifier confidence thresthold should be 0.7 based on the 20 files
('zh', 0.7384932637214661)
('zh', 0.9241222739219666)
('en', 0.8285728096961975)
('zh', 0.9961936473846436)
('zh', 0.9350615739822388)
('zh', 0.9060114026069641)
('zh', 0.959487795829773)
('zh', 0.9853195548057556)
('nl', 0.9027391076087952)
('el', 0.9998682141304016)
('el', 0.9998763799667358)
('zh', 0.9810582399368286)
('tr', 0.8735378980636597)
('en', 0.9264044165611267)
('da', 0.28138768672943115)
('zh', 0.9619793891906738)
('zh', 0.9739780426025391)
('zh', 0.9297752976417542)
('zh', 0.9969664812088013)
('zh', 0.964841902256012)
"""

def get_language(text: str) -> tuple[Any, float]:
    text = " ".join(text.split())
    model = fasttext.load_model("/Users/YangWen/Documents/Code/github/data/data/classifier/lid.176.bin")
    labels, probabilities = model.predict(text)
    lang = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    return (lang, confidence)


def extract_warc(file_name: str) -> None:
    stream = GZipStream(FileStream(file_name, 'rb'))
    cnt = 20
    cur = 0
    for record in ArchiveIterator(stream):
        if record.record_type == WarcRecordType.response:
            payload_bytes = record.reader.read()
            text = extract_text(payload_bytes)
            print(get_language(text))
            cur += 1
            if cur == cnt:
                break


# extract_warc('/Users/YangWen/Documents/Code/github/data/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz')
