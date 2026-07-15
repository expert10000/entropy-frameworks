from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from visionentropy.entropy.base import EntropyResult


def probability_distribution(
    data: ArrayLike,
    *,
    bins: int | None = None,
    assume_probabilities: bool = False,
) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("entropy input cannot be empty")

    if assume_probabilities:
        probabilities = values
    else:
        if bins is None:
            unique, counts = np.unique(values, return_counts=True)
            probabilities = counts.astype(np.float64)
        else:
            probabilities, _ = np.histogram(values, bins=bins)
            probabilities = probabilities.astype(np.float64)

    probabilities = probabilities[probabilities > 0]
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("probability distribution must have a positive sum")
    return probabilities / total


def _log(values: NDArray[np.float64], base: float) -> NDArray[np.float64]:
    if base <= 0 or base == 1:
        raise ValueError("logarithm base must be positive and not equal to 1")
    return np.log(values) / np.log(base)


@dataclass(frozen=True)
class ShannonEntropy:
    name: str = "shannon"

    def compute(
        self,
        data: ArrayLike,
        *,
        bins: int | None = None,
        assume_probabilities: bool = False,
        logarithm_base: float = 2.0,
        **_: Any,
    ) -> EntropyResult:
        p = probability_distribution(data, bins=bins, assume_probabilities=assume_probabilities)
        value = float(-(p * _log(p, logarithm_base)).sum())
        return EntropyResult(
            value=value,
            distribution=p,
            metadata={"bins": bins, "logarithm_base": logarithm_base},
        )


@dataclass(frozen=True)
class RenyiEntropy:
    alpha: float = 2.0
    name: str = "renyi"

    def compute(
        self,
        data: ArrayLike,
        *,
        bins: int | None = None,
        assume_probabilities: bool = False,
        logarithm_base: float = 2.0,
        **_: Any,
    ) -> EntropyResult:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if np.isclose(self.alpha, 1.0):
            return ShannonEntropy().compute(
                data,
                bins=bins,
                assume_probabilities=assume_probabilities,
                logarithm_base=logarithm_base,
            )

        p = probability_distribution(data, bins=bins, assume_probabilities=assume_probabilities)
        value = float(_log(np.array([(p**self.alpha).sum()]), logarithm_base)[0] / (1.0 - self.alpha))
        return EntropyResult(
            value=value,
            distribution=p,
            metadata={"alpha": self.alpha, "bins": bins, "logarithm_base": logarithm_base},
        )


@dataclass(frozen=True)
class TsallisEntropy:
    q: float = 2.0
    name: str = "tsallis"

    def compute(
        self,
        data: ArrayLike,
        *,
        bins: int | None = None,
        assume_probabilities: bool = False,
        **_: Any,
    ) -> EntropyResult:
        if self.q <= 0:
            raise ValueError("q must be positive")
        if np.isclose(self.q, 1.0):
            return ShannonEntropy().compute(
                data,
                bins=bins,
                assume_probabilities=assume_probabilities,
            )

        p = probability_distribution(data, bins=bins, assume_probabilities=assume_probabilities)
        value = float((1.0 - (p**self.q).sum()) / (self.q - 1.0))
        return EntropyResult(
            value=value,
            distribution=p,
            metadata={"q": self.q, "bins": bins},
        )
