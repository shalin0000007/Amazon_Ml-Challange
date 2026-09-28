import re
import math
import difflib
from typing import List, Dict, Any, Set, Tuple
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

DIGIT_RE = re.compile(r"\b\d+\b")
WORD_RE = re.compile(r"[a-z0-9]+")
VOWELS_RE = re.compile(r"[aeiouy]+")

FEATURE_NAMES = [
    # 1. Fine-grained Name Features (21)
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_min_overlap",
    "name_jaro_winkler",
    "name_jw_high_prefix",
    "name_word_jaccard",
    "name_word_dice",
    "first_word_match",
    "last_word_match",
    "name_tri_jaccard",
    "name_4gram_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_exact_match",
    "name_containment",
    "name_soundex_match",
    "name_consonant_ratio",
    "name_lcs_ratio",
    "name_token_diff",
    # 2. Hierarchical Address Features (11)
    "addr_token_sort",
    "addr_token_set",
    "addr_partial_ratio",
    "addr_word_jaccard",
    "addr_jaro_winkler",
    "addr_tri_jaccard",
    "addr_len_ratio",
    "addr_first_match",
    "addr_end_match",
    "addr_lcs_ratio",
    "addr_is_nan",
    # 3. Numeric & Postal Geography (6)
    "num_status",
    "num_conflict",
    "num_common_count",
    "zip_status",
    "zip_prefix_match",
    "suffix_match",
    # 4. Context & Ranking Metrics (5)
    "source_origin",
    "candidate_rank",
    "candidate_recip_rank",
    "is_top1",
    "is_top3",
    # 5. Composite Cross-Domain Interaction Features (4)
    "harmonic_name_addr",
    "geometric_name_addr",
    "strong_name_weak_addr",
    "strong_addr_weak_name",
]

# Soundex mapping table
SOUNDEX_TABLE = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}


def fast_soundex(word: str) -> str:
    """Computes standard 4-character American Soundex code."""
    if not word:
        return "0000"
    word = word.lower()
    first_char = word[0].upper()
    tail = word[1:]

    coded = []
    prev_code = SOUNDEX_TABLE.get(word[0], "")
    for char in tail:
        code = SOUNDEX_TABLE.get(char, "")
        if code != prev_code:
            if code:
                coded.append(code)
            prev_code = code

    res = first_char + "".join(coded)
    res = res.replace("0", "")
    return (res + "0000")[:4]


def longest_common_substring_len(s1: str, s2: str) -> int:
    """Computes exact length of longest contiguous common substring with fast C-path."""
    if not s1 or not s2:
        return 0
    if s1 in s2:
        return len(s1)
    if s2 in s1:
        return len(s2)
    return difflib.SequenceMatcher(None, s1, s2, autojunk=False).find_longest_match(0, len(s1), 0, len(s2)).size


def extract_zips(raw_addr: str) -> Set[str]:
    """Extracts 5-6 digit postal codes / PIN codes."""
    if not raw_addr or not isinstance(raw_addr, str) or str(raw_addr).lower() == "nan":
        return set()
    nums = DIGIT_RE.findall(str(raw_addr))
    return {n for n in nums if len(n) in (5, 6)}


def extract_pair_features(
    s1_name: str,
    s1_suffix: str,
    s1_addr: str,
    s1_nums: Set[str],
    s1_zips: Set[str],
    ext_id: str,
    ext_name: str,
    ext_suffix: str,
    ext_addr: str,
    ext_nums: Set[str],
    ext_zips: Set[str],
    rank: int,
) -> List[float]:
    """
    Computes a 47-dimensional high-discriminative pairwise feature vector for V4.
    Engineered to maximize Macro F0.5 by penalizing cross-store false merges.
    """

    # =========================================================================
    # 1. Fine-Grained Name Features
    # =========================================================================
    n_ratio = fuzz.ratio(s1_name, ext_name)
    n_partial = fuzz.partial_ratio(s1_name, ext_name)
    n_sort = fuzz.token_sort_ratio(s1_name, ext_name)
    n_set = fuzz.token_set_ratio(s1_name, ext_name)
    n_min_overlap = min(n_sort, n_set)
    n_jw = JaroWinkler.similarity(s1_name, ext_name) * 100.0
    n_jw_high = JaroWinkler.similarity(s1_name, ext_name, prefix_weight=0.15) * 100.0

    len1, len2 = len(s1_name), len(ext_name)
    max_nl = max(len1, len2)
    n_len_diff = float(abs(len1 - len2))
    n_len_ratio = (min(len1, len2) / max_nl) if max_nl > 0 else 0.0
    n_exact = 1.0 if s1_name and s1_name == ext_name else 0.0

    # Substring Containment
    if min(len1, len2) >= 4 and (s1_name in ext_name or ext_name in s1_name):
        n_containment = (min(len1, len2) / max_nl) * 100.0
    else:
        n_containment = 0.0

    # Word sets
    s1_words = s1_name.split()
    ext_words = ext_name.split()
    s1_word_set = set(s1_words)
    ext_word_set = set(ext_words)
    union_w = len(s1_word_set | ext_word_set)
    inter_w = len(s1_word_set & ext_word_set)
    n_word_jaccard = (inter_w / union_w * 100.0) if union_w > 0 else 0.0
    n_word_dice = (2.0 * inter_w / (len(s1_word_set) + len(ext_word_set)) * 100.0) if (len(s1_word_set) + len(ext_word_set)) > 0 else 0.0

    # First & Last Word Matches
    first_w_match = 1.0 if s1_words and ext_words and s1_words[0] == ext_words[0] else 0.0
    last_w_match = 1.0 if s1_words and ext_words and s1_words[-1] == ext_words[-1] else 0.0

    # Character Trigrams & 4-grams
    tri_s1 = {s1_name[i:i+3] for i in range(len1 - 2)} if len1 >= 3 else set()
    tri_ext = {ext_name[i:i+3] for i in range(len2 - 2)} if len2 >= 3 else set()
    union_tri = len(tri_s1 | tri_ext)
    n_tri_jaccard = (len(tri_s1 & tri_ext) / union_tri * 100.0) if union_tri > 0 else 0.0

    four_s1 = {s1_name[i:i+4] for i in range(len1 - 3)} if len1 >= 4 else set()
    four_ext = {ext_name[i:i+4] for i in range(len2 - 3)} if len2 >= 4 else set()
    union_four = len(four_s1 | four_ext)
    n_4gram_jaccard = (len(four_s1 & four_ext) / union_four * 100.0) if union_four > 0 else 0.0

    # Phonetic Soundex Match on Primary Word
    if s1_words and ext_words:
        n_soundex = 1.0 if fast_soundex(s1_words[0]) == fast_soundex(ext_words[0]) else 0.0
    else:
        n_soundex = 0.0

    # Consonant Skeleton Ratio
    s1_cons = VOWELS_RE.sub("", s1_name)
    ext_cons = VOWELS_RE.sub("", ext_name)
    n_cons_ratio = fuzz.ratio(s1_cons, ext_cons) if (s1_cons and ext_cons) else 0.0

    # Longest Common Substring Ratio
    lcs_name = longest_common_substring_len(s1_name, ext_name)
    n_lcs_ratio = (lcs_name / max_nl * 100.0) if max_nl > 0 else 0.0

    # Token Count Difference
    n_tok_diff = float(abs(len(s1_words) - len(ext_words)))

    # =========================================================================
    # 2. Hierarchical Address Features
    # =========================================================================
    if s1_addr and ext_addr:
        a_sort = fuzz.token_sort_ratio(s1_addr, ext_addr)
        a_set = fuzz.token_set_ratio(s1_addr, ext_addr)
        a_partial = fuzz.partial_ratio(s1_addr, ext_addr)
        a_jw = JaroWinkler.similarity(s1_addr, ext_addr) * 100.0

        a_w1 = set(s1_addr.split())
        a_w2 = set(ext_addr.split())
        a_union = len(a_w1 | a_w2)
        a_jaccard = (len(a_w1 & a_w2) / a_union * 100.0) if a_union > 0 else 0.0

        al1, al2 = len(s1_addr), len(ext_addr)
        max_al = max(al1, al2)
        a_len_ratio = (min(al1, al2) / max_al) if max_al > 0 else 0.0

        # Tri-jaccard of address
        a_tri1 = {s1_addr[i:i+3] for i in range(al1 - 2)} if al1 >= 3 else set()
        a_tri2 = {ext_addr[i:i+3] for i in range(al2 - 2)} if al2 >= 3 else set()
        a_u_tri = len(a_tri1 | a_tri2)
        a_tri_jaccard = (len(a_tri1 & a_tri2) / a_u_tri * 100.0) if a_u_tri > 0 else 0.0

        s1_first_a = s1_addr.split()[0] if s1_addr.split() else ""
        ext_first_a = ext_addr.split()[0] if ext_addr.split() else ""
        a_first_match = 1.0 if s1_first_a and s1_first_a == ext_first_a else 0.0

        s1_end_a = s1_addr.split()[-1] if s1_addr.split() else ""
        ext_end_a = ext_addr.split()[-1] if ext_addr.split() else ""
        a_end_match = 1.0 if s1_end_a and s1_end_a == ext_end_a else 0.0

        lcs_addr = longest_common_substring_len(s1_addr, ext_addr)
        a_lcs_ratio = (lcs_addr / max_al * 100.0) if max_al > 0 else 0.0
        a_nan = 0.0
    else:
        a_sort = a_set = a_partial = a_jaccard = a_jw = a_tri_jaccard = a_len_ratio = 0.0
        a_first_match = a_end_match = a_lcs_ratio = 0.0
        a_nan = 1.0 if not ext_addr else 0.0

    # =========================================================================
    # 3. Numeric & Postal Geography
    # =========================================================================
    if s1_nums and ext_nums:
        common_nums = s1_nums & ext_nums
        num_common_count = float(len(common_nums))
        if common_nums:
            num_status = 1.0
            num_conflict = 0.0
        else:
            num_status = -1.0
            num_conflict = 1.0
    else:
        num_status = 0.0
        num_conflict = 0.0
        num_common_count = 0.0

    # Postal Code / PIN code verification
    if s1_zips and ext_zips:
        if s1_zips & ext_zips:
            zip_status = 1.0
            zip_prefix_match = 1.0
        else:
            zip_status = -1.0
            # Check 3-digit prefix (same postal district/region)
            s1_pre = {z[:3] for z in s1_zips}
            ext_pre = {z[:3] for z in ext_zips}
            zip_prefix_match = 1.0 if (s1_pre & ext_pre) else -1.0
    else:
        zip_status = 0.0
        zip_prefix_match = 0.0

    # Legal Suffix Match
    if s1_suffix and ext_suffix:
        suff_match = 1.0 if s1_suffix == ext_suffix else 0.0
    else:
        suff_match = -1.0

    # =========================================================================
    # 4. Context & Ranking Metrics
    # =========================================================================
    src_origin = 0.0 if ext_id.startswith("S2-") else 1.0
    cand_rank = float(rank)
    cand_recip_rank = 1.0 / (1.0 + float(rank))
    is_top1 = 1.0 if rank == 0 else 0.0
    is_top3 = 1.0 if rank < 3 else 0.0

    # =========================================================================
    # 5. Composite Cross-Domain Interaction Features
    # =========================================================================
    # Harmonic mean of name and address scores
    n_score = max(n_sort, n_set)
    a_score = max(a_sort, a_set)
    harmonic = (2.0 * n_score * a_score / (n_score + a_score + 1e-5))
    geometric = math.sqrt(n_score * a_score)

    # Chain store detector (name matches, but completely different address)
    strong_name_weak_addr = 1.0 if (n_score >= 88.0 and a_score < 25.0) else 0.0

    # Shopping mall / Co-located business detector (address matches, but totally different name)
    strong_addr_weak_name = 1.0 if (a_score >= 88.0 and n_score < 25.0) else 0.0

    return [
        # Name
        n_ratio, n_partial, n_sort, n_set, n_min_overlap,
        n_jw, n_jw_high, n_word_jaccard, n_word_dice,
        first_w_match, last_w_match, n_tri_jaccard, n_4gram_jaccard,
        n_len_diff, n_len_ratio, n_exact, n_containment,
        n_soundex, n_cons_ratio, n_lcs_ratio, n_tok_diff,
        # Address
        a_sort, a_set, a_partial, a_jaccard, a_jw,
        a_tri_jaccard, a_len_ratio, a_first_match, a_end_match,
        a_lcs_ratio, a_nan,
        # Numbers & Zips
        num_status, num_conflict, num_common_count,
        zip_status, zip_prefix_match, suff_match,
        # Context & Rank
        src_origin, cand_rank, cand_recip_rank, is_top1, is_top3,
        # Composites
        harmonic, geometric, strong_name_weak_addr, strong_addr_weak_name
    ]
