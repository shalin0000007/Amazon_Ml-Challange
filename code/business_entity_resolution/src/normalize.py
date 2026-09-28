import re
import unicodedata
from typing import Tuple, List, Set

try:
    from anyascii import anyascii
except ImportError:
    anyascii = lambda x: x

# Precompute translation table for maximum speed (500k+ strings/sec)
PUNCT_SYMB_MAP = {i: " " for i in range(0x10000) if unicodedata.category(chr(i)).startswith(("P", "S"))}
LATIN_FOLD_MAP = {}
for i in range(0x10000):
    ch = chr(i)
    if "LATIN" in unicodedata.name(ch, ""):
        decomp = unicodedata.normalize("NFKD", ch)
        base = "".join(c for c in decomp if not unicodedata.combining(c))
        if base != ch:
            LATIN_FOLD_MAP[i] = base

COMBINED_TRANSLATE_TABLE = {**PUNCT_SYMB_MAP, **LATIN_FOLD_MAP}
DIGIT_RE = re.compile(r"\b\d+\b")

# Legal suffixes to identify and standardize
LEGAL_SUFFIXES = {
    # English / Universal
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "pvt", "private", "llc", "llp", "gmbh", "ent", "enterprises",
    # French
    "sa", "sarl", "sas", "sasu", "snc", "eurl", "cie", "ets",
    # Hindi / Devanagari / South Indian common words & transliterations
    "प्राइवेट", "लिमिटेड", "एलएलपी", "praiveta", "limiteda", "limited", "praiveta", "praivet", "limiteDa", "teknAlajIs", "enterpraises",
    "லிமிடெட்", "பிரைவேட்", "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್", "কোম্পানি", "প্রাইভেট", "লিমিটেড"
}

# Common address abbreviations (English, French, Hindi)
ADDRESS_ABBR = {
    # English
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "hwy": "highway",
    "pkwy": "parkway",
    "opp": "opposite",
    "nr": "near",
    # French
    "bd": "boulevard",
    "bvd": "boulevard",
    "av": "avenue",
    "all": "allee",
    "imp": "impasse",
    "che": "chemin",
    "rte": "route",
    "r": "rue",
    "crs": "cours",
    "qu": "quai",
}

STOP_WORDS = {"and", "the", "for", "of", "in", "to", "at", "by", "from", "&"}


TAMIL_FIXES = {
    'ன': 'n', 'ற': 'r', 'ழ': 'zh', 'ள': 'l', 'ண': 'n', 'ஸ': 's', 'ஷ': 'sh', 'ஜ': 'j', 'ஹ': 'h'
}

LEET_0_RE = re.compile(r"([a-zA-Z])0([a-zA-Z])")
LEET_1_RE = re.compile(r"([a-zA-Z])1([a-zA-Z])")


def is_indic_text(text: str) -> bool:
    """Checks if text contains Indic unicode characters."""
    for ch in text:
        if "\u0900" <= ch <= "\u0d7f":
            return True
    return False


SCRIPT_RANGES = [
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
]


def romanize_indic_text(text: str) -> str:
    """Ultra-fast transliteration of Indic script strings to Latin phonetics with targeted script detection."""
    if not is_indic_text(text):
        return text
    for k, v in TAMIL_FIXES.items():
        if k in text:
            text = text.replace(k, v)
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        target_scripts = set()
        for c in text:
            code = ord(c)
            if 0x0900 <= code <= 0x0D7F:
                for lo, hi, scr in SCRIPT_RANGES:
                    if lo <= code <= hi:
                        target_scripts.add(getattr(sanscript, scr.upper(), None))
                        break
        for scr in target_scripts:
            if scr:
                text = transliterate(text, scr, sanscript.ITRANS)
    except Exception:
        pass
    return text


CAMEL_RE = re.compile(r"([a-z])([A-Z])")
DOT_EXT_RE = re.compile(r"\.(com|org|net|in|co|io|fr|ai|biz)\b", re.IGNORECASE)
PREFIX_TITLES_RE = re.compile(r"^(m/s|mr\b|ms\b|dr\b|shri\b|smt\b)\s*", re.IGNORECASE)
DIGIT_ALPHA_RE = re.compile(r"(\d+)([a-zA-Z]+)")
ALPHA_DIGIT_RE = re.compile(r"([a-zA-Z]+)(\d+)")
BIZ_NOUNS_RE = re.compile(
    r"(projects|engineering|enterprises|solutions|technologies|services|industries|remedies|systems|ventures|pioneer|foundation|institute)",
    re.IGNORECASE,
)


def clean_text_fast(text: str) -> str:
    """Ultra-fast multilingual normalization with sub-word, domain, and title segmentation."""
    if not text or not isinstance(text, str) or text.lower() == "nan":
        return ""
    # Strip honorific / title prefixes (e.g. M/s, Mr, Dr)
    text = PREFIX_TITLES_RE.sub("", text.strip())
    # Decode leetspeak digits in words (e.g. vzh0tels -> vzhotels)
    text = LEET_0_RE.sub(r"\1o\2", text)
    text = LEET_1_RE.sub(r"\1i\2", text)
    # Segment domain extensions and camelCase words
    text = DOT_EXT_RE.sub(" ", text)
    text = CAMEL_RE.sub(r"\1 \2", text)
    # Segment merged digits and letters (e.g. 401bhagwansingh -> 401 bhagwansingh)
    text = DIGIT_ALPHA_RE.sub(r"\1 \2", text)
    text = ALPHA_DIGIT_RE.sub(r"\1 \2", text)
    # Segment common concatenated business nouns
    text = BIZ_NOUNS_RE.sub(r" \1 ", text)
    # Universal transliteration of Indic, non-Latin, and accented European scripts
    text = anyascii(text)
    return " ".join(text.translate(COMBINED_TRANSLATE_TABLE).lower().split())


def normalize_business_name(raw_name: str) -> Tuple[str, str, List[str]]:
    """Normalizes business name, transliterates Indic scripts, and isolates legal suffixes."""
    clean = clean_text_fast(raw_name)
    tokens = clean.split()
    if not tokens:
        return "", "", []

    core_tokens = []
    found_suffixes = []

    for t in tokens:
        if t in LEGAL_SUFFIXES:
            found_suffixes.append(t)
        else:
            core_tokens.append(t)

    suffix_str = " ".join(found_suffixes)
    core_name = " ".join(core_tokens) if core_tokens else clean
    distinctive = [t for t in core_tokens if len(t) >= 2 and t not in STOP_WORDS]

    return core_name, suffix_str, distinctive


def normalize_address(raw_address: str) -> Tuple[str, Set[str], List[str]]:
    """Normalizes address string, transliterates Indic addresses, and extracts numerical components."""
    clean = clean_text_fast(raw_address)
    tokens = clean.split()
    expanded_tokens = [ADDRESS_ABBR.get(t, t) for t in tokens]
    clean_expanded = " ".join(expanded_tokens)

    numbers = {n.lstrip("0") or "0" for n in DIGIT_RE.findall(clean)}
    return clean_expanded, numbers, expanded_tokens
