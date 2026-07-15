import numpy as np
import pytest

from visionentropy.entropy.information import RenyiEntropy, ShannonEntropy, TsallisEntropy


def test_shannon_uniform_distribution() -> None:
    result = ShannonEntropy().compute([0.25, 0.25, 0.25, 0.25], assume_probabilities=True)

    assert result.value == pytest.approx(2.0)


def test_shannon_degenerate_distribution() -> None:
    result = ShannonEntropy().compute([1.0, 0.0, 0.0, 0.0], assume_probabilities=True)

    assert result.value == pytest.approx(0.0)


def test_renyi_near_one_matches_shannon() -> None:
    distribution = np.array([0.1, 0.2, 0.3, 0.4])

    shannon = ShannonEntropy().compute(distribution, assume_probabilities=True).value
    renyi = RenyiEntropy(alpha=1.0).compute(distribution, assume_probabilities=True).value

    assert renyi == pytest.approx(shannon)


def test_tsallis_near_one_matches_shannon() -> None:
    distribution = np.array([0.1, 0.2, 0.3, 0.4])

    shannon = ShannonEntropy().compute(distribution, assume_probabilities=True).value
    tsallis = TsallisEntropy(q=1.0).compute(distribution, assume_probabilities=True).value

    assert tsallis == pytest.approx(shannon)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        ShannonEntropy().compute([])
