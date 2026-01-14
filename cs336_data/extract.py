"""
command
uv run pytest -k test_extract_text_from_html_bytes
uv run pytest -k test_identify_language
uv run pytest -k test_mask_emails
uv run pytest -k test_mask_phones
uv run pytest -k test_mask_ips
uv run pytest -k test_classify_nsfw
uv run pytest -k test_classify_toxic_speech
uv run pytest -k test_gopher
"""

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

from typing import Any, List
import re

from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext


def extract_text(html_bytes: bytes) -> str | None:
    try:
        encoding = detect_encoding(html_bytes)
        bytes_str = html_bytes.decode(encoding or "utf-8", errors="replace")
        text = extract_plain_text(bytes_str)
        return text if text else None
    except Exception:
        return None


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


"""
mask_pii
4
replacing diverse forms into a single placeholder token collapses many distinct contexts into one
use typed but diverse placeholders, randomize placeholders slightly

long placeholder strings introduce unnatrual token patterns and sequence lengths
replace with short consistent tokens, add them explicitly to the tokenizer vocabulary

5
false positive
population 2001219099
this is replaced by |||PHONE_NUMBER|||

false negative
email tom@white
"""


EMAIL_PATTERN = re.compile(
    r"""
    (?<![\w.-])                # left boundary
    [a-zA-Z0-9._%+-]+           # username
    @
    [a-zA-Z0-9.-]+              # domain
    \.[a-zA-Z]{2,}              # TLD
    (?![\w.-])                  # right boundary
    """,
    re.VERBOSE
)

def mask_email(text: str) -> tuple[str, int]:
    return EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)


PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)                          # no digit before
    (?:\+?1[\s.-]?)?                 # optional country code
    (?:                              # area code
        \(?\d{3}\)?                  # (123) or 123
        [\s.-]?
    )
    \d{3}                            # exchange
    [\s.-]?
    \d{4}                            # line number
    (?!\d)                           # no digit after
    """,
    re.VERBOSE
)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_PATTERN.subn("|||PHONE_NUMBER|||", text)


IPV4_PATTERN = re.compile(
    r"""
    (?<!\d)                                   # no digit before
    (?:                                      # first three octets
        (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)   # 0–255
        \.
    ){3}
    (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)       # last octet
    (?!\d)                                   # no digit after
    """,
    re.VERBOSE
)


def mask_ips(text: str) -> tuple[str, int]:
    return IPV4_PATTERN.subn("|||IP_ADDRESS|||", text)


"""
harmful_content
3.
over sanitization
aggressive filtering removes a large fraction of content, including borderline or contextually safe text

model can fial to respond to sensitive topcis

loss of realistic context
model may underestimate social context cues

mitigation strategies
use probabilistic / soft filtering
partial masking instead of removal

4.
fraction is about 30%
classifier error
false positive, a website about Apache, it is classified as NSFW and toxic speed
threshold should be 0.9
"""

def classify_NSFW(text: str) -> tuple[Any, float]:
    text = " ".join(text.split())
    model = fasttext.load_model("/Users/YangWen/Documents/Code/github/data/data/classifier/jigsaw_fasttext_bigrams_nsfw_final.bin")
    labels, probabilities = model.predict(text)
    lang = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    return (lang, confidence)

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    text = " ".join(text.split())
    model = fasttext.load_model("/Users/YangWen/Documents/Code/github/data/data/classifier/jigsaw_fasttext_bigrams_hatespeech_final.bin")
    labels, probabilities = model.predict(text)
    lang = labels[0].replace("__label__", "")
    confidence = probabilities[0]
    return (lang, confidence)


"""
gopher_quality_filters
(b)
harmful information might be classified as high quality
repeat the same thing multiple times is classifed as hgih quality
"""


def has_alpha(word: str) -> bool:
    """Check if a word contains at least one alphabetic character."""
    return any(c.isalpha() for c in word)


def gopher_quality_filter(text: str) -> bool:
    """
    Returns True if the document passes all filters,
    False if it should be removed.
    """

    words = re.findall(r"\b\w+\b", text)
    num_words = len(words)

    if num_words < 50 or num_words > 100_000:
        return False

    mean_word_length = sum(len(w) for w in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.endswith("...") for line in lines)
        if ellipsis_lines / len(lines) > 0.30:
            return False

    alpha_words = sum(has_alpha(w) for w in words)
    if alpha_words / num_words < 0.80:
        return False

    return True


def extract_warc(file_name: str) -> None:
    stream = GZipStream(FileStream(file_name, 'rb'))
    cnt = 20
    cur = 0
    for record in ArchiveIterator(stream):
        if record.record_type == WarcRecordType.response:
            payload_bytes = record.reader.read()
            text = extract_text(payload_bytes)
            if text is None:
                continue
            print(gopher_quality_filter(text), ' '.join(text.split()))
            # _, confidence1 = classify_NSFW(text)
            # _, confidence2 = classify_toxic_speech(text)
            # print(text, confidence1, confidence2)
            cur += 1
            if cur == cnt:
                break

# extract_warc('/Users/YangWen/Documents/Code/github/data/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz')
