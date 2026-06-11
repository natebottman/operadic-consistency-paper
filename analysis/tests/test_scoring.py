"""Smoke tests for analysis/scoring.py.

Run with: python -m pytest analysis/tests/test_scoring.py -v
"""
import math
from analysis.scoring import (
    normalize, first_number, f1, score, is_correct,
    pairwise_f1, cluster_entropy,
    squad_normalize, squad_f1, drop_score, extract_yes_no, strategyqa_score,
    score_official, is_correct_official,
)


def test_normalize_basic():
    assert normalize("Hello, World!") == "hello world"
    assert normalize("The cat") == "cat"
    assert normalize("113,532") == "113532"
    assert normalize("Five cities.") == "5 cities"


def test_first_number():
    assert first_number("5") == 5.0
    assert first_number("Five cities.") == 5.0
    assert first_number("12.5") == 12.5            # decimal preserved
    assert first_number("8-yard TD pass") == 8.0   # hyphen splits compound
    assert first_number("113,532") == 113532.0     # comma inside number
    assert first_number("no number here") is None


def test_f1_perfect_match():
    assert f1("hello world", "hello world") == 1.0


def test_f1_no_overlap():
    assert f1("hello", "world") == 0.0


def test_score_number_fallback_match():
    # Both have numbers, equal -> 1.0 even if surface forms differ.
    # Verbose-correct case: leading "Two" -> 2 wins over compound "1-yard".
    assert score("Two 1-yard touchdowns.", "2") == 1.0
    assert score("five", "5") == 1.0
    assert score("12.5", "12.5") == 1.0


def test_score_number_fallback_mismatch():
    # Both have standalone numbers, unequal -> 0.0
    assert score("4 cars", "5 cars") == 0.0
    assert score("4", "5") == 0.0
    # Compound forms with mismatched leading numbers: also 0.0
    assert score("8-yard TD pass", "3-yard TD pass") == 0.0


def test_score_text_fallback():
    # Neither side has a number -> falls back to token-F1
    assert score("Sydney Harbour", "Sydney Harbour") == 1.0
    assert 0.0 < score("Sydney Harbour bridge", "Sydney Harbour") < 1.0


def test_is_correct_with_aliases():
    assert is_correct("Coast Range", "the Coast Range") is True
    assert is_correct("Coast Range", "Mountains", aliases=["the Coast Range"]) is True
    assert is_correct("", "anything") is False
    assert is_correct("anything", None) is False


def test_pairwise_f1():
    # Three identical samples -> 1.0
    assert pairwise_f1(["a b", "a b", "a b"]) == 1.0
    # Three disjoint samples -> 0.0
    assert pairwise_f1(["a", "b", "c"]) == 0.0


def test_cluster_entropy_unanimous():
    # All same cluster -> entropy 0
    assert cluster_entropy(["paris", "paris", "paris"]) == 0.0


def test_cluster_entropy_split():
    # Two clusters of equal size -> entropy log(2)
    h = cluster_entropy(["paris", "paris", "london", "london"])
    assert abs(h - math.log(2)) < 1e-6


# ── Per-dataset official scorers ────────────────────────────────────────────

def test_squad_normalize():
    assert squad_normalize("Hello, World!") == "hello world"
    assert squad_normalize("The cat") == "cat"
    # SQuAD normalize does NOT do word-to-digit
    assert squad_normalize("five cities") == "five cities"


def test_squad_f1():
    assert squad_f1("hello world", "hello world") == 1.0
    assert squad_f1("hello", "world") == 0.0
    # Word-form numbers are NOT collapsed by SQuAD F1
    assert squad_f1("five", "5") == 0.0


def test_drop_score_number_match():
    # Number-equality fallback applies on numerical pairs
    assert drop_score("5", "5") == 1.0
    assert drop_score("Two 1-yard touchdowns.", "2") == 1.0
    assert drop_score("4", "5") == 0.0


def test_drop_score_text_fallback():
    # No numbers on either side -> SQuAD F1
    assert drop_score("Sydney Harbour", "Sydney Harbour") == 1.0


def test_extract_yes_no():
    assert extract_yes_no("Yes") == "yes"
    assert extract_yes_no("No.") == "no"
    assert extract_yes_no("Yes, X is Y") == "yes"
    assert extract_yes_no("True") == "yes"      # 'true' -> 'yes'
    assert extract_yes_no("false") == "no"
    assert extract_yes_no("It depends.") is None
    assert extract_yes_no("") is None


def test_strategyqa_score():
    assert strategyqa_score("Yes", "yes") == 1.0
    assert strategyqa_score("Yes, X is Y", "yes") == 1.0   # hedging not punished
    assert strategyqa_score("No", "yes") == 0.0
    assert strategyqa_score("Maybe", "yes") == 0.0          # no extraction -> 0


def test_score_official_dispatch():
    # SQuAD-style for HotpotQA / MuSiQue
    assert score_official("five", "5", "hotpotqa") == 0.0  # no word-to-digit
    # DROP: number fallback active
    assert score_official("5", "5", "drop") == 1.0
    # StrategyQA: yes/no extraction
    assert score_official("Yes, X", "yes", "strategyqa") == 1.0
    assert score_official("No", "yes", "strategyqa") == 0.0
    # GSM8K: same as DROP (number-equality)
    assert score_official("answer is 42", "42", "gsm8k") == 1.0


def test_is_correct_official():
    assert is_correct_official("Yes, X is Y", "yes", "strategyqa") is True
    assert is_correct_official("No", "yes", "strategyqa") is False
    # Aliases respected
    assert is_correct_official("the Coast Range", "Coastal mountains",
                                "hotpotqa", aliases=["Coast Range"]) is True
