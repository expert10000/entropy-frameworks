from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DeepEntropyResult:
    model_name: str
    feature_map: NDArray[np.float32]
    activation_entropy: NDArray[np.float32]
    latent_vector: NDArray[np.float32]
    logits: NDArray[np.float32]
    predictive_probabilities: NDArray[np.float32]
    latent_entropy: float
    predictive_entropy: float
    mean_activation_entropy: float
    available: bool = True
    message: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "message": self.message,
            "model": self.model_name,
            "featureShape": list(self.feature_map.shape),
            "latentSize": int(self.latent_vector.size),
            "classCount": int(self.predictive_probabilities.size),
            "meanActivationEntropy": self.mean_activation_entropy,
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
    image_size: int = 224,
    random_state: int = 0,
) -> DeepEntropyResult:
    try:
        import torch
        from torchvision.models import resnet18
    except Exception as error:  # noqa: BLE001 - optional deep stack.
        return _unavailable_result(model_name=model_name, message=str(error))

    if model_name != "resnet18":
        raise ValueError("Stage 4 currently supports model_name='resnet18'")

    torch.manual_seed(random_state)
    model = resnet18(weights=None)
    model.eval()

    tensor = _image_tensor(image, image_size=image_size, torch=torch)
    with torch.no_grad():
        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        feature_tensor = model.layer4(x)
        pooled = model.avgpool(feature_tensor)
        latent_tensor = torch.flatten(pooled, 1)
        logits_tensor = model.fc(latent_tensor)
        probabilities_tensor = torch.softmax(logits_tensor, dim=1)

    feature_map = feature_tensor.detach().cpu().numpy()[0].astype(np.float32)
    latent_vector = latent_tensor.detach().cpu().numpy()[0].astype(np.float32)
    logits = logits_tensor.detach().cpu().numpy()[0].astype(np.float32)
    probabilities = probabilities_tensor.detach().cpu().numpy()[0].astype(np.float32)
    activation_entropy = activation_entropy_map(feature_map)
    return DeepEntropyResult(
        model_name=model_name,
        feature_map=feature_map,
        activation_entropy=activation_entropy,
        latent_vector=latent_vector,
        logits=logits,
        predictive_probabilities=probabilities,
        latent_entropy=distribution_entropy(np.abs(latent_vector)),
        predictive_entropy=distribution_entropy(probabilities),
        mean_activation_entropy=float(activation_entropy.mean()) if activation_entropy.size else 0.0,
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


def _unavailable_result(*, model_name: str, message: str) -> DeepEntropyResult:
    empty_feature = np.zeros((0, 0, 0), dtype=np.float32)
    empty_vector = np.zeros(0, dtype=np.float32)
    return DeepEntropyResult(
        model_name=model_name,
        feature_map=empty_feature,
        activation_entropy=np.zeros((0, 0), dtype=np.float32),
        latent_vector=empty_vector,
        logits=empty_vector,
        predictive_probabilities=empty_vector,
        latent_entropy=0.0,
        predictive_entropy=0.0,
        mean_activation_entropy=0.0,
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
