from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.draw import disk, ellipse, rectangle
from skimage.filters import gaussian
from skimage.util import random_noise

from visionentropy.datasets.base import ImageSample


@dataclass(frozen=True)
class SyntheticShapesConfig:
    image_size: tuple[int, int] = (256, 256)
    sample_count: int = 16
    shape_count: int | None = None
    min_shapes: int = 2
    max_shapes: int = 5
    foreground_texture: float = 0.0
    background_texture: float = 0.0
    gaussian_noise: float = 0.0
    impulse_noise: float = 0.0
    noise_amount: float | None = None
    boundary_blur: float = 0.7
    illumination_gradient: float = 0.0
    allow_overlap: bool = False
    contrast: float = 1.0
    preset: str | None = None
    seed: int = 42


SYNTHETIC_BENCHMARK_PRESETS: dict[str, dict[str, object]] = {
    "s01_clean_high_contrast": {
        "shape_count": 3,
        "contrast": 1.15,
        "boundary_blur": 0.5,
        "illumination_gradient": 0.05,
        "allow_overlap": False,
        "seed": 101,
    },
    "s02_gaussian_noise": {
        "shape_count": 3,
        "gaussian_noise": 0.08,
        "contrast": 1.0,
        "boundary_blur": 0.6,
        "allow_overlap": False,
        "seed": 102,
    },
    "s03_impulse_noise": {
        "shape_count": 3,
        "impulse_noise": 0.06,
        "contrast": 1.0,
        "boundary_blur": 0.5,
        "allow_overlap": False,
        "seed": 103,
    },
    "s04_blurred_boundaries": {
        "shape_count": 3,
        "boundary_blur": 2.2,
        "contrast": 1.0,
        "allow_overlap": False,
        "seed": 104,
    },
    "s05_textured_foreground": {
        "shape_count": 3,
        "foreground_texture": 0.35,
        "contrast": 1.0,
        "boundary_blur": 0.6,
        "allow_overlap": False,
        "seed": 105,
    },
    "s06_textured_background": {
        "shape_count": 3,
        "background_texture": 0.35,
        "contrast": 1.0,
        "boundary_blur": 0.6,
        "allow_overlap": False,
        "seed": 106,
    },
    "s07_overlapping_objects": {
        "shape_count": 5,
        "contrast": 1.0,
        "boundary_blur": 0.6,
        "allow_overlap": True,
        "seed": 107,
    },
    "s08_low_contrast": {
        "shape_count": 3,
        "contrast": 0.42,
        "boundary_blur": 0.6,
        "illumination_gradient": 0.08,
        "allow_overlap": False,
        "seed": 108,
    },
}


def synthetic_preset_names() -> list[str]:
    return list(SYNTHETIC_BENCHMARK_PRESETS)


def synthetic_config_from_preset(
    preset: str,
    *,
    image_size: tuple[int, int] = (256, 256),
    sample_count: int = 16,
) -> SyntheticShapesConfig:
    if preset not in SYNTHETIC_BENCHMARK_PRESETS:
        raise ValueError(f"Unknown synthetic preset: {preset}")
    return SyntheticShapesConfig(
        image_size=image_size,
        sample_count=sample_count,
        preset=preset,
        **SYNTHETIC_BENCHMARK_PRESETS[preset],
    )


class SyntheticShapesDataset:
    """Procedural dataset for deterministic segmentation experiments."""

    def __init__(self, config: SyntheticShapesConfig | None = None) -> None:
        self.config = config or SyntheticShapesConfig()

    def __len__(self) -> int:
        return self.config.sample_count

    def __getitem__(self, index: int) -> ImageSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        rng = np.random.default_rng(self.config.seed + index)
        height, width = self.config.image_size
        image = self._background(rng, height, width)
        mask = np.zeros((height, width), dtype=np.uint8)

        shape_count = self.config.shape_count
        if shape_count is None:
            shape_count = int(rng.integers(self.config.min_shapes, self.config.max_shapes + 1))
        for shape_id in range(1, shape_count + 1):
            color = np.clip(0.18 + (rng.uniform(0.35, 0.95, size=3) * self.config.contrast), 0.0, 1.0)
            rr, cc = self._draw_non_overlapping_shape(rng, height, width, mask)
            image[rr, cc] = color
            if self.config.foreground_texture > 0:
                texture = self._texture(rng, height, width, self.config.foreground_texture)
                image[rr, cc] = np.clip(image[rr, cc] + texture[rr, cc, None], 0.0, 1.0)
            mask[rr, cc] = shape_id

        if self.config.illumination_gradient > 0:
            image = self._apply_illumination_gradient(image, rng)
        if self.config.boundary_blur > 0:
            image = gaussian(
                image,
                sigma=float(self.config.boundary_blur),
                channel_axis=-1,
                preserve_range=True,
            )
        if self.config.gaussian_noise > 0:
            image = image + rng.normal(0.0, self.config.gaussian_noise, size=image.shape)
        impulse_noise = self.config.impulse_noise
        if self.config.noise_amount is not None:
            impulse_noise = self.config.noise_amount
        if impulse_noise > 0:
            image = random_noise(image, mode="s&p", amount=impulse_noise, rng=self.config.seed + index)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        return ImageSample(
            sample_id=f"synthetic_{index:04d}",
            image=image,
            mask=mask,
            metadata={
                "source": "synthetic_shapes",
                "preset": self.config.preset,
                "shape_count": shape_count,
                "foreground_texture": self.config.foreground_texture,
                "background_texture": self.config.background_texture,
                "gaussian_noise": self.config.gaussian_noise,
                "impulse_noise": impulse_noise,
                "boundary_blur": self.config.boundary_blur,
                "illumination_gradient": self.config.illumination_gradient,
                "allow_overlap": self.config.allow_overlap,
                "contrast": self.config.contrast,
                "seed": self.config.seed,
            },
        )

    def _background(self, rng: np.random.Generator, height: int, width: int) -> NDArray[np.float32]:
        base = np.full((height, width, 3), 0.08, dtype=np.float32)
        if self.config.background_texture > 0:
            texture = self._texture(rng, height, width, self.config.background_texture)
            base = np.clip(base + texture[..., None], 0.0, 1.0)
        return base.astype(np.float32)

    def _draw_non_overlapping_shape(
        self,
        rng: np.random.Generator,
        height: int,
        width: int,
        mask: NDArray[np.uint8],
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        last_shape = self._draw_shape(rng.choice(["disk", "ellipse", "rectangle"]), rng, height, width)
        if self.config.allow_overlap:
            return last_shape
        for _ in range(24):
            shape_kind = rng.choice(["disk", "ellipse", "rectangle"])
            rr, cc = self._draw_shape(shape_kind, rng, height, width)
            last_shape = (rr, cc)
            if not np.any(mask[rr, cc]):
                return rr, cc
        return last_shape

    @staticmethod
    def _texture(
        rng: np.random.Generator,
        height: int,
        width: int,
        amount: float,
    ) -> NDArray[np.float32]:
        noise = rng.normal(0.0, amount, size=(height, width)).astype(np.float32)
        coarse = gaussian(noise, sigma=max(1.0, min(height, width) / 40.0), preserve_range=True)
        return np.asarray(coarse, dtype=np.float32)

    def _apply_illumination_gradient(
        self,
        image: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        height, width = image.shape[:2]
        if rng.random() < 0.5:
            ramp = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
        else:
            ramp = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
        factor = 1.0 + (self.config.illumination_gradient * ramp)
        return np.clip(image * factor[..., None], 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _draw_shape(
        shape_kind: str,
        rng: np.random.Generator,
        height: int,
        width: int,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        center_r = int(rng.integers(height // 5, height - height // 5))
        center_c = int(rng.integers(width // 5, width - width // 5))
        radius_r = int(rng.integers(max(8, height // 14), max(12, height // 5)))
        radius_c = int(rng.integers(max(8, width // 14), max(12, width // 5)))

        if shape_kind == "disk":
            return disk((center_r, center_c), min(radius_r, radius_c), shape=(height, width))

        if shape_kind == "ellipse":
            return ellipse(center_r, center_c, radius_r, radius_c, shape=(height, width))

        start = (max(0, center_r - radius_r), max(0, center_c - radius_c))
        extent = (min(height - start[0], radius_r * 2), min(width - start[1], radius_c * 2))
        return rectangle(start=start, extent=extent, shape=(height, width))
