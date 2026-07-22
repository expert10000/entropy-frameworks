from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


@dataclass(frozen=True)
class DeepEntropyResult:
    model_name: str
    layer_name: str
    representation_level: str
    uncertainty_method: str
    feature_map: NDArray[np.float32]
    activation_entropy: NDArray[np.float32]
    fuzzy_entropy: NDArray[np.float32]
    rough_uncertainty: NDArray[np.float32]
    fuzzy_rough_uncertainty: NDArray[np.float32]
    latent_vector: NDArray[np.float32]
    logits: NDArray[np.float32]
    predictive_probabilities: NDArray[np.float32]
    latent_entropy: float
    predictive_entropy: float
    mean_activation_entropy: float
    mean_fuzzy_entropy: float
    mean_rough_uncertainty: float
    mean_fuzzy_rough_uncertainty: float
    neighborhood_k: int
    similarity_sigma: float
    available: bool = True
    message: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "message": self.message,
            "model": self.model_name,
            "layer": self.layer_name,
            "representationLevel": self.representation_level,
            "uncertaintyMethod": self.uncertainty_method,
            "featureShape": list(self.feature_map.shape),
            "latentSize": int(self.latent_vector.size),
            "classCount": int(self.predictive_probabilities.size),
            "neighborhoodK": self.neighborhood_k,
            "similaritySigma": self.similarity_sigma,
            "meanActivationEntropy": self.mean_activation_entropy,
            "meanFuzzyEntropy": self.mean_fuzzy_entropy,
            "meanRoughUncertainty": self.mean_rough_uncertainty,
            "meanFuzzyRoughUncertainty": self.mean_fuzzy_rough_uncertainty,
            "latentEntropy": self.latent_entropy,
            "predictiveEntropy": self.predictive_entropy,
            "topProbabilities": [
                {"index": int(index), "probability": float(self.predictive_probabilities[index])}
                for index in np.argsort(self.predictive_probabilities)[-8:][::-1]
            ],
        }


def deep_learning_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def analyze_deep_features(
    image: NDArray[np.generic],
    *,
    model_name: str = "resnet18",
    layer_name: str = "layer4",
    representation_level: str = "pixel_embedding",
    uncertainty_method: str = "classical",
    image_size: int = 224,
    neighborhood_k: int = 15,
    similarity_sigma: float | None = None,
    random_state: int = 0,
) -> DeepEntropyResult:
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - optional deep stack.
        return _unavailable_result(
            model_name=model_name,
            layer_name=layer_name,
            representation_level=representation_level,
            uncertainty_method=uncertainty_method,
            neighborhood_k=neighborhood_k,
            similarity_sigma=similarity_sigma or 0.0,
            message=str(error),
        )

    torch.manual_seed(random_state)
    tensor = _image_tensor(image, image_size=image_size, torch=torch)
    if model_name == "small_cnn":
        feature_tensor, latent_tensor, logits_tensor = _small_cnn_forward(
            tensor,
            layer_name=layer_name,
            torch=torch,
        )
    elif model_name in {"resnet18", "resnet34"}:
        try:
            from torchvision.models import resnet18, resnet34
        except Exception as error:  # noqa: BLE001 - optional deep stack.
            return _unavailable_result(
                model_name=model_name,
                layer_name=layer_name,
                representation_level=representation_level,
                uncertainty_method=uncertainty_method,
                neighborhood_k=neighborhood_k,
                similarity_sigma=similarity_sigma or 0.0,
                message=str(error),
            )
        builder = resnet18 if model_name == "resnet18" else resnet34
        model = builder(weights=None)
        model.eval()
        feature_tensor, latent_tensor, logits_tensor = _resnet_forward(
            model,
            tensor,
            layer_name=layer_name,
            torch=torch,
        )
    else:
        raise ValueError("Deep entropy supports model_name in {'small_cnn', 'resnet18', 'resnet34'}")

    probabilities_tensor = torch.softmax(logits_tensor, dim=1)

    feature_map = feature_tensor.detach().cpu().numpy()[0].astype(np.float32)
    latent_vector = latent_tensor.detach().cpu().numpy()[0].astype(np.float32)
    logits = logits_tensor.detach().cpu().numpy()[0].astype(np.float32)
    probabilities = probabilities_tensor.detach().cpu().numpy()[0].astype(np.float32)
    activation_entropy = activation_entropy_map(feature_map)
    similarity_sigma_value = _resolved_sigma(feature_map, similarity_sigma)
    uncertainty = deep_uncertainty_maps(
        feature_map,
        neighborhood_k=neighborhood_k,
        similarity_sigma=similarity_sigma_value,
        random_state=random_state,
    )
    return DeepEntropyResult(
        model_name=model_name,
        layer_name=layer_name,
        representation_level=representation_level,
        uncertainty_method=uncertainty_method,
        feature_map=feature_map,
        activation_entropy=activation_entropy,
        fuzzy_entropy=uncertainty["fuzzy_entropy"],
        rough_uncertainty=uncertainty["rough_uncertainty"],
        fuzzy_rough_uncertainty=uncertainty["fuzzy_rough_uncertainty"],
        latent_vector=latent_vector,
        logits=logits,
        predictive_probabilities=probabilities,
        latent_entropy=distribution_entropy(np.abs(latent_vector)),
        predictive_entropy=distribution_entropy(probabilities),
        mean_activation_entropy=float(activation_entropy.mean()) if activation_entropy.size else 0.0,
        mean_fuzzy_entropy=float(uncertainty["fuzzy_entropy"].mean()) if uncertainty["fuzzy_entropy"].size else 0.0,
        mean_rough_uncertainty=float(uncertainty["rough_uncertainty"].mean())
        if uncertainty["rough_uncertainty"].size
        else 0.0,
        mean_fuzzy_rough_uncertainty=float(uncertainty["fuzzy_rough_uncertainty"].mean())
        if uncertainty["fuzzy_rough_uncertainty"].size
        else 0.0,
        neighborhood_k=int(neighborhood_k),
        similarity_sigma=float(similarity_sigma_value),
    )


def activation_projection(feature_map: NDArray[np.generic]) -> NDArray[np.float32]:
    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature_map must have shape (channels, height, width)")
    return _zero_one(np.mean(np.abs(features), axis=0))


def activation_entropy_map(feature_map: NDArray[np.generic]) -> NDArray[np.float32]:
    features = np.abs(np.asarray(feature_map, dtype=np.float32))
    if features.ndim != 3:
        raise ValueError("feature_map must have shape (channels, height, width)")
    channel_count = features.shape[0]
    total = features.sum(axis=0, keepdims=True)
    probabilities = np.divide(features, total, out=np.zeros_like(features), where=total > 0)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))).sum(axis=0)
    if channel_count > 1:
        entropy /= np.log(channel_count)
    return entropy.astype(np.float32)


def deep_uncertainty_maps(
    feature_map: NDArray[np.generic],
    *,
    neighborhood_k: int = 15,
    similarity_sigma: float | None = None,
    random_state: int = 0,
) -> dict[str, NDArray[np.float32]]:
    embeddings, height, width = _spatial_embeddings(feature_map)
    if embeddings.shape[0] == 0:
        empty = np.zeros((height, width), dtype=np.float32)
        return {
            "fuzzy_entropy": empty,
            "rough_uncertainty": empty,
            "fuzzy_rough_uncertainty": empty,
        }

    sigma = _resolved_sigma(feature_map, similarity_sigma)
    memberships = fuzzy_memberships(embeddings, sigma=sigma, random_state=random_state)
    fuzzy_entropy = normalized_entropy(memberships).reshape(height, width)
    rough_uncertainty, fuzzy_rough_uncertainty = rough_uncertainties(
        embeddings,
        memberships,
        neighborhood_k=neighborhood_k,
        sigma=sigma,
    )
    return {
        "fuzzy_entropy": fuzzy_entropy.astype(np.float32),
        "rough_uncertainty": rough_uncertainty.reshape(height, width).astype(np.float32),
        "fuzzy_rough_uncertainty": fuzzy_rough_uncertainty.reshape(height, width).astype(np.float32),
    }


def fuzzy_memberships(
    embeddings: NDArray[np.generic],
    *,
    sigma: float,
    random_state: int,
    prototype_count: int = 3,
) -> NDArray[np.float32]:
    normalized = _standardize_rows(np.asarray(embeddings, dtype=np.float32))
    sample_count = normalized.shape[0]
    cluster_count = max(2, min(prototype_count, sample_count))
    if sample_count <= 1:
        return np.ones((sample_count, 1), dtype=np.float32)
    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init="auto")
    model.fit(normalized)
    distances = pairwise_distances(normalized, model.cluster_centers_, metric="euclidean")
    scale = max(float(sigma), 1e-6)
    similarities = np.exp(-(distances * distances) / (2.0 * scale * scale))
    totals = similarities.sum(axis=1, keepdims=True)
    return np.divide(similarities, totals, out=np.zeros_like(similarities), where=totals > 0).astype(np.float32)


def rough_uncertainties(
    embeddings: NDArray[np.generic],
    memberships: NDArray[np.generic],
    *,
    neighborhood_k: int,
    sigma: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    normalized = _standardize_rows(np.asarray(embeddings, dtype=np.float32))
    membership_array = np.asarray(memberships, dtype=np.float32)
    sample_count = normalized.shape[0]
    if sample_count <= 1:
        return np.zeros(sample_count, dtype=np.float32), np.zeros(sample_count, dtype=np.float32)

    distances = pairwise_distances(normalized, metric="euclidean")
    neighbor_count = max(1, min(int(neighborhood_k), sample_count - 1))
    neighbor_indices = np.argsort(distances, axis=1)[:, 1 : neighbor_count + 1]
    labels = np.argmax(membership_array, axis=1)
    rough = np.zeros(sample_count, dtype=np.float32)
    fuzzy_rough = np.zeros(sample_count, dtype=np.float32)
    scale = max(float(sigma), 1e-6)

    for index in range(sample_count):
        neighbors = neighbor_indices[index]
        if neighbors.size == 0:
            continue
        same_class = labels[neighbors] == labels[index]
        rough[index] = 1.0 - float(same_class.mean())

        similarities = np.exp(-(distances[index, neighbors] ** 2) / (2.0 * scale * scale))
        class_memberships = membership_array[neighbors, labels[index]]
        total_similarity = float(similarities.sum())
        if total_similarity > 0:
            fuzzy_rough[index] = float(
                np.clip(
                    (similarities * (1.0 - class_memberships)).sum() / total_similarity,
                    0.0,
                    1.0,
                )
            )
    return rough, fuzzy_rough


def normalized_entropy(probabilities: NDArray[np.generic]) -> NDArray[np.float32]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] <= 1:
        return np.zeros(p.shape[0], dtype=np.float32)
    entropy = -(p * np.log(np.clip(p, 1e-9, 1.0))).sum(axis=1)
    return (entropy / np.log(p.shape[1])).astype(np.float32)


def distribution_entropy(values: NDArray[np.generic]) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[array > 0]
    if array.size == 0:
        return 0.0
    probabilities = array / array.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / np.log(probabilities.size) if probabilities.size > 1 else 0.0


def predictive_entropy_map(
    predictive_entropy: float,
    shape: tuple[int, int],
) -> NDArray[np.float32]:
    return np.full(shape, float(predictive_entropy), dtype=np.float32)


def _image_tensor(image: NDArray[np.generic], *, image_size: int, torch: Any) -> Any:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    array = _zero_one(array)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _resnet_forward(model: Any, tensor: Any, *, layer_name: str, torch: Any) -> tuple[Any, Any, Any]:
    with torch.no_grad():
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        captures = {"stem": x}
        x = model.maxpool(x)
        x = model.layer1(x)
        captures["layer1"] = x
        x = model.layer2(x)
        captures["layer2"] = x
        x = model.layer3(x)
        captures["layer3"] = x
        x = model.layer4(x)
        captures["layer4"] = x
        pooled = model.avgpool(x)
        latent = torch.flatten(pooled, 1)
        logits = model.fc(latent)
        captures["avgpool"] = pooled
        captures["logits"] = logits.unsqueeze(-1).unsqueeze(-1)
    if layer_name not in captures:
        raise ValueError("ResNet layer must be one of stem, layer1, layer2, layer3, layer4, avgpool, logits")
    return captures[layer_name], latent, logits


def _small_cnn_forward(tensor: Any, *, layer_name: str, torch: Any) -> tuple[Any, Any, Any]:
    nn = torch.nn
    stem = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
    layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
    layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    classifier = nn.Linear(64, 8)
    modules = [stem, layer1, layer2, avgpool, classifier]
    for module in modules:
        module.eval()
    with torch.no_grad():
        x = stem(tensor)
        captures = {"stem": x}
        x = layer1(x)
        captures["layer1"] = x
        x = layer2(x)
        captures["layer2"] = x
        pooled = avgpool(x)
        latent = torch.flatten(pooled, 1)
        logits = classifier(latent)
        captures["avgpool"] = pooled
        captures["logits"] = logits.unsqueeze(-1).unsqueeze(-1)
    if layer_name not in captures:
        raise ValueError("Small CNN layer must be one of stem, layer1, layer2, avgpool, logits")
    return captures[layer_name], latent, logits


def _spatial_embeddings(feature_map: NDArray[np.generic]) -> tuple[NDArray[np.float32], int, int]:
    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature_map must have shape (channels, height, width)")
    _, height, width = features.shape
    embeddings = features.transpose(1, 2, 0).reshape(height * width, features.shape[0])
    return embeddings.astype(np.float32), height, width


def _standardize_rows(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    return np.divide(array - mean, std, out=np.zeros_like(array), where=std > 1e-6).astype(np.float32)


def _resolved_sigma(feature_map: NDArray[np.generic], configured_sigma: float | None) -> float:
    if configured_sigma is not None and configured_sigma > 0:
        return float(configured_sigma)
    embeddings, _, _ = _spatial_embeddings(feature_map)
    if embeddings.shape[0] <= 1:
        return 1.0
    normalized = _standardize_rows(embeddings)
    sample = normalized[: min(256, normalized.shape[0])]
    distances = pairwise_distances(sample, metric="euclidean")
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1.0
    return float(np.clip(np.median(positive), 0.05, 5.0))


def _unavailable_result(
    *,
    model_name: str,
    layer_name: str,
    representation_level: str,
    uncertainty_method: str,
    neighborhood_k: int,
    similarity_sigma: float,
    message: str,
) -> DeepEntropyResult:
    empty_feature = np.zeros((0, 0, 0), dtype=np.float32)
    empty_vector = np.zeros(0, dtype=np.float32)
    return DeepEntropyResult(
        model_name=model_name,
        layer_name=layer_name,
        representation_level=representation_level,
        uncertainty_method=uncertainty_method,
        feature_map=empty_feature,
        activation_entropy=np.zeros((0, 0), dtype=np.float32),
        fuzzy_entropy=np.zeros((0, 0), dtype=np.float32),
        rough_uncertainty=np.zeros((0, 0), dtype=np.float32),
        fuzzy_rough_uncertainty=np.zeros((0, 0), dtype=np.float32),
        latent_vector=empty_vector,
        logits=empty_vector,
        predictive_probabilities=empty_vector,
        latent_entropy=0.0,
        predictive_entropy=0.0,
        mean_activation_entropy=0.0,
        mean_fuzzy_entropy=0.0,
        mean_rough_uncertainty=0.0,
        mean_fuzzy_rough_uncertainty=0.0,
        neighborhood_k=neighborhood_k,
        similarity_sigma=similarity_sigma,
        available=False,
        message=message,
    )


def _zero_one(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)
