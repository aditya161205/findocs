"""Unit tests for citation + confidence logic (no API key or network required)."""

from src.rag_engine import Answer, Source, _confidence


def _src(number: int, relevance: float) -> Source:
    return Source(
        number=number,
        source="annual_report.pdf",
        page=number,
        relevance=relevance,
        text=f"passage {number}",
    )


def test_confidence_empty_is_zero():
    assert _confidence([]) == 0.0


def test_confidence_high_when_top_match_strong():
    score = _confidence([_src(1, 0.95), _src(2, 0.9)])
    assert score > 0.75


def test_confidence_low_when_matches_weak():
    score = _confidence([_src(1, 0.30), _src(2, 0.25)])
    assert score < 0.5


def test_confidence_is_bounded():
    assert 0.0 <= _confidence([_src(1, 1.0), _src(2, 1.0)]) <= 1.0


def test_confidence_label_thresholds():
    assert Answer(text="x", confidence=0.9).confidence_label == "High"
    assert Answer(text="x", confidence=0.6).confidence_label == "Medium"
    assert Answer(text="x", confidence=0.3).confidence_label == "Low"


def test_confidence_pct_rounds():
    assert Answer(text="x", confidence=0.834).confidence_pct == 83


def test_source_label_with_and_without_page():
    assert Source(1, "f.pdf", 4, 0.9, "t").label == "f.pdf (p.4)"
    assert Source(1, "f.pdf", None, 0.9, "t").label == "f.pdf"
