import pytest

from genksr.training.schedules import warmup_cosine_factor


def test_warmup_cosine_factor_hits_requested_endpoints():
    min_factor = 1e-7 / 1e-3

    assert warmup_cosine_factor(0, 10, 2, min_factor) == pytest.approx(min_factor)
    assert warmup_cosine_factor(2, 10, 2, min_factor) == pytest.approx(1.0)
    assert warmup_cosine_factor(10, 10, 2, min_factor) == pytest.approx(min_factor)
    assert warmup_cosine_factor(20, 10, 2, min_factor) == pytest.approx(min_factor)


def test_warmup_cosine_factor_without_warmup_starts_at_base_lr():
    assert warmup_cosine_factor(0, 10, 0, 1e-4) == pytest.approx(1.0)
    assert warmup_cosine_factor(10, 10, 0, 1e-4) == pytest.approx(1e-4)


def test_warmup_cosine_factor_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        warmup_cosine_factor(0, 10, 10, 1e-4)
    with pytest.raises(ValueError):
        warmup_cosine_factor(0, 10, 2, 1.1)
