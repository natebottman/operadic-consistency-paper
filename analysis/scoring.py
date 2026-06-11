"""Per-dataset scoring primitives matching each benchmark's official protocol.

All paper numbers are computed against `score_official` / `is_correct_official`,
which dispatch on dataset:
  - HotpotQA, MuSiQue: SQuAD-style token-F1 (lowercase, strip articles, strip
    punctuation, whitespace-fix; standard SQuAD evaluation script).
  - DROP: number-equality fallback when both sides have a parseable number,
    otherwise SQuAD-style token-F1. Approximates DROP's official scorer for
    its dominant numerical-answer case.
  - StrategyQA: extract the first standalone yes/no/true/false token from
    each side and exact-match. Matches the standard StrategyQA boolean
    evaluation protocol.

The legacy uniform `score` / `is_correct` (token-F1 + number-fallback +
word-to-digit) is retained for use by the SC and -H sample-based scorers
where a uniform metric across datasets is appropriate.
"""
import math
import re
import string
from collections import Counter

WORD_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
    'thousand': 1000,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'eleventh': 11, 'twelfth': 12,
}


def normalize(text):
    """Lowercase, strip articles/punct, collapse digit commas, convert spelled numbers."""
    t = str(text).lower().strip()
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    t = re.sub(r'(\d),(\d)', r'\1\2', t)
    t = re.sub(r'[^a-z0-9 ]', '', t)
    return ' '.join(str(WORD_NUM[k]) if k in WORD_NUM else k for k in t.split()).strip()


def first_number(text):
    """Extract first standalone integer or decimal.

    Handles three patterns that pure normalize-then-regex misses:
      (a) Decimals like "12.5" (normalize strips '.', leaving "125").
      (b) Hyphenated compounds like "8-yard" (normalize strips '-', joining
          to "8yard" which has no word-boundary on the right of the digits).
      (c) Word-form leading numbers like "Two 1-yard touchdowns." -> 2,
          where word-to-digit conversion ("Two" -> "2") needs to win over
          the compound's "1".

    Strategy: check decimals against the raw text first (preserves the dot),
    then split hyphens/slashes and run normalize (which converts spelled-out
    numbers to digits and collapses commas inside numbers), then take the
    first standalone integer.
    """
    if text is None:
        return None
    s = str(text)
    # (a) Decimals must be matched against raw text -- normalize strips '.'.
    m = re.search(r'(?<!\w)(\d+\.\d+)(?!\w)', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # (b)/(c) Split compound separators, then normalize (handles word->digit,
    # digit-comma collapse), then take the first standalone integer.
    s_split = re.sub(r'[-/]', ' ', s)
    norm = normalize(s_split)
    m = re.search(r'\b(\d+)\b', norm)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def f1(pred, gold):
    """Token-F1 after v1 normalization."""
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr = n / len(p)
    re_ = n / len(g)
    return 2 * pr * re_ / (pr + re_)


def score(pred, gold):
    """v1 score: number-fallback if both sides have a parseable number, else token-F1."""
    pn = first_number(pred)
    gn = first_number(gold)
    if pn is not None and gn is not None:
        return 1.0 if pn == gn else 0.0
    return f1(pred, gold)


def is_correct(pred, gold, aliases=None, threshold=0.5):
    """Indicator that pred matches gold (or any alias) at score >= threshold."""
    if not pred or not str(pred).strip() or gold is None:
        return False
    candidates = [gold] + list(aliases or [])
    return max(score(pred, c) for c in candidates) >= threshold


def pairwise_f1(samples):
    """Mean pairwise token-F1 over all unordered sample pairs.

    Legacy: dataset-agnostic. Prefer `pairwise_official(samples, dataset)` so
    that pairwise comparisons use the same equivalence as accuracy and OC.
    """
    if len(samples) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total += f1(samples[i], samples[j])
            count += 1
    return total / count if count else 0.0


def pairwise_official(samples, dataset):
    """Mean pairwise score under score_official(_, _, dataset).

    Matched-axis SC: the pairwise comparator equals the accuracy/OC scorer for
    the dataset (yes/no extraction for StrategyQA, value-equality for DROP/GSM8K,
    SQuAD-F1 elsewhere). See App.~B of the paper for motivation.
    """
    if len(samples) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            total += score_official(samples[i], samples[j], dataset)
            count += 1
    return total / count if count else 0.0


def cluster_entropy(samples, threshold=0.5, dataset=None):
    """Greedy-cluster samples by agreement >= threshold, return Shannon entropy of cluster sizes.

    When `dataset` is supplied, agreement is the matched-axis per-dataset
    `score_official(s, rep, dataset)` (yes/no extraction for StrategyQA,
    value-equality for DROP/GSM8K, SQuAD-F1 elsewhere). When omitted, falls
    back to dataset-agnostic `f1` (legacy behavior).
    """
    if not samples:
        return 0.0
    cmp = (lambda s, rep: score_official(s, rep, dataset)) if dataset is not None else f1
    clusters = []
    for s in samples:
        placed = False
        for i, (rep, cnt) in enumerate(clusters):
            if cmp(s, rep) >= threshold:
                clusters[i] = (rep, cnt + 1)
                placed = True
                break
        if not placed:
            clusters.append((s, 1))
    K = len(samples)
    h = 0.0
    for _, c in clusters:
        p = c / K
        if p > 0:
            h -= p * math.log(p)
    return h


# ── SQuAD-style scorer (HotpotQA, MuSiQue) ──────────────────────────────────

def squad_normalize(text):
    """Standard SQuAD eval normalize: lowercase, strip articles, strip punct."""
    if text is None:
        return ''
    t = str(text).lower()
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    t = ''.join(c for c in t if c not in string.punctuation)
    return ' '.join(t.split())


def squad_f1(pred, gold):
    """SQuAD-style token-F1. No number-fallback, no word-to-digit conversion."""
    p = squad_normalize(pred).split()
    g = squad_normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr = n / len(p)
    re_ = n / len(g)
    return 2 * pr * re_ / (pr + re_)


# ── DROP-style scorer (number-equality fallback + SQuAD F1) ─────────────────

def drop_score(pred, gold):
    """Number-equality if both sides have a parseable number; SQuAD F1 otherwise."""
    pn = first_number(pred)
    gn = first_number(gold)
    if pn is not None and gn is not None:
        return 1.0 if pn == gn else 0.0
    return squad_f1(pred, gold)


# ── StrategyQA-style scorer (yes/no extraction + EM) ────────────────────────

def extract_yes_no(text):
    """Extract first standalone yes/no/true/false. Returns 'yes', 'no', or None.

    'true' maps to 'yes', 'false' maps to 'no'.
    """
    if not text:
        return None
    m = re.search(r'\b(yes|no|true|false)\b', str(text).lower())
    if not m:
        return None
    return 'yes' if m.group(1) in ('yes', 'true') else 'no'


def strategyqa_score(pred, gold):
    """1.0 if both pred and gold extract to the same yes/no, else 0.0."""
    p = extract_yes_no(pred)
    g = extract_yes_no(gold)
    if p is None or g is None:
        return 0.0
    return 1.0 if p == g else 0.0


# ── Per-dataset dispatcher ──────────────────────────────────────────────────

def score_official(pred, gold, dataset):
    """Official-protocol score for the given dataset.

    Used for both correctness (pred vs gold) and OC (direct vs decomposed).
    """
    if dataset == 'strategyqa':
        return strategyqa_score(pred, gold)
    if dataset in ('drop', 'gsm8k'):
        # Both rely on number-equality for the dominant numerical-answer case.
        return drop_score(pred, gold)
    # HotpotQA, MuSiQue (and any unknown dataset): SQuAD-style F1.
    return squad_f1(pred, gold)


def is_correct_official(pred, gold, dataset, aliases=None, threshold=0.5):
    """Indicator that pred matches gold (or any alias) under the per-dataset
    official scorer at score >= threshold."""
    if not pred or not str(pred).strip() or gold is None:
        return False
    candidates = [gold] + list(aliases or [])
    return max(score_official(pred, c, dataset) for c in candidates) >= threshold
