from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.segmentation import slic


@dataclass(frozen=True)
class RegionRepresentation:
    labels: NDArray[np.int32]
    stats: list[dict[str, float | int]]
    edges: list[tuple[int, int]]

    @property
    def region_count(self) -> int:
        return len(self.stats)


def build_region_representation(
    image: NDArray[np.generic],
    *,
    intensity: NDArray[np.generic],
    entropy_map: NDArray[np.generic],
    n_segments: int = 96,
    compactness: float = 12.0,
    sigma: float = 0.5,
    entropy_bins: int = 32,
) -> RegionRepresentation:
    labels = slic(
        _zero_one(np.asarray(image, dtype=np.float32)),
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1 if np.asarray(image).ndim == 3 else None,
    ).astype(np.int32)
    stats = region_statistics(
        labels,
        intensity=_zero_one(intensity),
        entropy_map=np.asarray(entropy_map, dtype=np.float32),
        entropy_bins=entropy_bins,
    )
    edges = region_adjacency(labels)
    neighbor_counts = {index: 0 for index in range(len(stats))}
    for left, right in edges:
        neighbor_counts[left] += 1
        neighbor_counts[right] += 1
    for row in stats:
        row["neighbor_count"] = neighbor_counts[int(row["label"])]
    return RegionRepresentation(labels=labels, stats=stats, edges=edges)


def region_statistics(
    labels: NDArray[np.int32],
    *,
    intensity: NDArray[np.generic],
    entropy_map: NDArray[np.generic],
    entropy_bins: int = 32,
) -> list[dict[str, float | int]]:
    label_array = np.asarray(labels, dtype=np.int32)
    intensity_array = np.asarray(intensity, dtype=np.float32)
    entropy_array = np.asarray(entropy_map, dtype=np.float32)
    if intensity_array.shape != label_array.shape or entropy_array.shape != label_array.shape:
        raise ValueError("region statistic inputs must match the label map shape")

    y_coordinates, x_coordinates = np.indices(label_array.shape)
    rows: list[dict[str, float | int]] = []
    for label in range(int(label_array.max()) + 1):
        mask = label_array == label
        if not np.any(mask):
            continue
        values = intensity_array[mask]
        entropy_values = entropy_array[mask]
        rows.append(
            {
                "label": int(label),
                "pixel_count": int(mask.sum()),
                "centroid_y": float(y_coordinates[mask].mean()),
                "centroid_x": float(x_coordinates[mask].mean()),
                "mean_intensity": float(values.mean()),
                "std_intensity": float(values.std()),
                "mean_entropy": float(entropy_values.mean()),
                "region_entropy": distribution_entropy(values, bins=entropy_bins),
                "neighbor_count": 0,
            }
        )
    return rows


def region_adjacency(labels: NDArray[np.int32]) -> list[tuple[int, int]]:
    label_array = np.asarray(labels, dtype=np.int32)
    edge_set: set[tuple[int, int]] = set()
    for first, second in (
        (label_array[:, :-1], label_array[:, 1:]),
        (label_array[:-1, :], label_array[1:, :]),
    ):
        changed = first != second
        if not np.any(changed):
            continue
        pairs = np.stack([first[changed], second[changed]], axis=-1)
        for left, right in pairs:
            a, b = sorted((int(left), int(right)))
            edge_set.add((a, b))
    return sorted(edge_set)


def region_value_image(
    labels: NDArray[np.int32],
    stats: list[dict[str, float | int]],
    field: str,
) -> NDArray[np.float32]:
    label_array = np.asarray(labels, dtype=np.int32)
    values = np.zeros(int(label_array.max()) + 1, dtype=np.float32)
    for row in stats:
        values[int(row["label"])] = float(row[field])
    return values[label_array]


def region_graph_payload(representation: RegionRepresentation) -> dict[str, Any]:
    return {
        "regionCount": representation.region_count,
        "edgeCount": len(representation.edges),
        "nodes": representation.stats,
        "edges": [{"source": int(source), "target": int(target)} for source, target in representation.edges],
    }


def distribution_entropy(values: NDArray[np.generic], *, bins: int) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0 or np.isclose(float(array.min()), float(array.max())):
        return 0.0
    histogram, _ = np.histogram(array, bins=bins, range=(float(array.min()), float(array.max())))
    probabilities = histogram.astype(np.float64)
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probabilities /= total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def _zero_one(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)
