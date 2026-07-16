from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering

from visionentropy.representations import RegionRepresentation


@dataclass(frozen=True)
class GraphEntropyResult:
    node_entropy: NDArray[np.float32]
    edge_weights: NDArray[np.float32]
    edge_entropy: NDArray[np.float32]
    eigenvalues: NDArray[np.float32]
    partition_labels: NDArray[np.int32]
    spectral_entropy: float
    normalized_spectral_entropy: float
    mean_node_entropy: float
    mean_edge_entropy: float

    def payload(self, regions: RegionRepresentation) -> dict[str, Any]:
        nodes = []
        for index, row in enumerate(regions.stats):
            nodes.append(
                {
                    "label": int(row["label"]),
                    "node_entropy": float(self.node_entropy[index]),
                    "partition": int(self.partition_labels[index]),
                }
            )
        edges = []
        for index, (source, target) in enumerate(regions.edges):
            edges.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "weight": float(self.edge_weights[index]),
                    "edge_entropy": float(self.edge_entropy[index]),
                }
            )
        return {
            "meanNodeEntropy": self.mean_node_entropy,
            "meanEdgeEntropy": self.mean_edge_entropy,
            "spectralEntropy": self.spectral_entropy,
            "normalizedSpectralEntropy": self.normalized_spectral_entropy,
            "partitionCount": int(np.unique(self.partition_labels).size),
            "nodes": nodes,
            "edges": edges,
            "eigenvalues": [float(value) for value in self.eigenvalues],
        }


def analyze_region_graph(
    regions: RegionRepresentation,
    *,
    partition_count: int = 2,
    random_state: int = 0,
) -> GraphEntropyResult:
    node_entropy = _node_entropy(regions)
    adjacency, edge_weights = _weighted_adjacency(regions, node_entropy)
    edge_entropy = _edge_entropy(edge_weights)
    eigenvalues, spectral_entropy, normalized_spectral_entropy = _spectral_entropy(adjacency)
    partition_labels = _graph_partitions(
        adjacency,
        partition_count=partition_count,
        random_state=random_state,
    )
    return GraphEntropyResult(
        node_entropy=node_entropy.astype(np.float32),
        edge_weights=edge_weights.astype(np.float32),
        edge_entropy=edge_entropy.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
        partition_labels=partition_labels.astype(np.int32),
        spectral_entropy=spectral_entropy,
        normalized_spectral_entropy=normalized_spectral_entropy,
        mean_node_entropy=float(node_entropy.mean()) if node_entropy.size else 0.0,
        mean_edge_entropy=float(edge_entropy.mean()) if edge_entropy.size else 0.0,
    )


def _node_entropy(regions: RegionRepresentation) -> NDArray[np.float32]:
    values = []
    for row in regions.stats:
        values.append(float(row["region_entropy"]) + float(row["mean_entropy"]))
    return _zero_one(np.asarray(values, dtype=np.float32))


def _weighted_adjacency(
    regions: RegionRepresentation,
    node_entropy: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    node_count = regions.region_count
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    edge_weights = np.zeros(len(regions.edges), dtype=np.float32)
    mean_intensity = np.asarray([float(row["mean_intensity"]) for row in regions.stats], dtype=np.float32)
    mean_entropy = np.asarray([float(row["mean_entropy"]) for row in regions.stats], dtype=np.float32)
    for index, (source, target) in enumerate(regions.edges):
        intensity_distance = abs(float(mean_intensity[source] - mean_intensity[target]))
        entropy_distance = abs(float(mean_entropy[source] - mean_entropy[target]))
        uncertainty_distance = abs(float(node_entropy[source] - node_entropy[target]))
        weight = np.exp(
            -3.0 * intensity_distance
            -2.0 * entropy_distance
            -1.25 * uncertainty_distance
        )
        edge_weights[index] = float(weight)
        adjacency[source, target] = float(weight)
        adjacency[target, source] = float(weight)
    return adjacency, edge_weights


def _edge_entropy(edge_weights: NDArray[np.float32]) -> NDArray[np.float32]:
    if edge_weights.size == 0:
        return edge_weights.astype(np.float32)
    probabilities = np.clip(edge_weights.astype(np.float64), 1e-9, 1.0 - 1e-9)
    entropy = -(probabilities * np.log(probabilities) + (1.0 - probabilities) * np.log(1.0 - probabilities))
    return (entropy / np.log(2.0)).astype(np.float32)


def _spectral_entropy(adjacency: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, float]:
    node_count = adjacency.shape[0]
    if node_count == 0:
        return np.zeros(0, dtype=np.float32), 0.0, 0.0
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inverse_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    normalized = adjacency * inverse_sqrt_degree[:, None] * inverse_sqrt_degree[None, :]
    laplacian = np.eye(node_count, dtype=np.float32) - normalized
    eigenvalues = np.maximum(np.linalg.eigvalsh(laplacian).astype(np.float64), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return eigenvalues.astype(np.float32), 0.0, 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalized_entropy = entropy / np.log(node_count) if node_count > 1 else 0.0
    return eigenvalues.astype(np.float32), entropy, float(normalized_entropy)


def _graph_partitions(
    adjacency: NDArray[np.float32],
    *,
    partition_count: int,
    random_state: int,
) -> NDArray[np.int32]:
    node_count = adjacency.shape[0]
    if node_count < 2:
        return np.zeros(node_count, dtype=np.int32)
    count = max(2, min(partition_count, node_count))
    affinity = adjacency.copy()
    np.fill_diagonal(affinity, 1.0)
    try:
        labels = SpectralClustering(
            n_clusters=count,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        ).fit_predict(affinity)
    except ValueError:
        labels = np.zeros(node_count, dtype=np.int32)
    return labels.astype(np.int32)


def _zero_one(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)
