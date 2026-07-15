from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.transform import resize

from visionentropy.datasets.base import ImageSample


@dataclass(frozen=True)
class ResizeSample:
    height: int = 256
    width: int = 256
    name: str = "resize"

    def transform(self, sample: ImageSample) -> ImageSample:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("resize height and width must be positive")

        image = resize(
            sample.image,
            output_shape=(self.height, self.width, *sample.image.shape[2:]),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        mask = None
        if sample.mask is not None:
            mask = resize(
                sample.mask,
                output_shape=(self.height, self.width),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(sample.mask.dtype)

        metadata = {
            **sample.metadata,
            "resize": {"height": self.height, "width": self.width},
        }
        return ImageSample(
            sample_id=sample.sample_id,
            image=image,
            mask=mask,
            label=sample.label,
            metadata=metadata,
        )
