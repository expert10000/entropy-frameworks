from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visionentropy.datasets.base import ImageSample


@dataclass(frozen=True)
class OxfordPetsConfig:
    root: Path = Path("data/raw/oxford_iiit_pet")
    split: str = "trainval"
    download: bool = False


class OxfordPetsDataset:
    """Thin wrapper around torchvision's Oxford-IIIT Pet dataset.

    The actual dataset is intentionally user-managed because the files are too
    large for the repository. Set ``download=True`` only in an explicit runner.
    """

    def __init__(self, config: OxfordPetsConfig | None = None) -> None:
        self.config = config or OxfordPetsConfig()
        try:
            from torchvision.datasets import OxfordIIITPet
        except ImportError as error:
            raise RuntimeError(
                "OxfordPetsDataset requires the optional deep dependencies: "
                'install with pip install -e ".[deep]".'
            ) from error

        self._dataset = OxfordIIITPet(
            root=str(self.config.root),
            split=self.config.split,
            target_types=("segmentation", "category"),
            download=self.config.download,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> ImageSample:
        image, target = self._dataset[index]
        mask, label = target
        return ImageSample(
            sample_id=f"oxford_pet_{index:05d}",
            image=np.asarray(image, dtype=np.float32) / 255.0,
            mask=np.asarray(mask),
            label=label,
            metadata={"source": "oxford_iiit_pet", "split": self.config.split},
        )
