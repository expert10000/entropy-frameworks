from __future__ import annotations

import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import yaml
from skimage.io import imsave

from visionentropy.datasets.registry import dataset_status, load_dataset_specs
from visionentropy.datasets.synthetic_shapes import SYNTHETIC_BENCHMARK_PRESETS
from visionentropy.pipeline import run_baseline_entropy_comparison, run_vertical_slice
from visionentropy.pipeline.metadata import build_run_metadata
from visionentropy.pipeline.vertical_slice import _load_sample, _preprocess_sample, _to_uint8_rgb, _to_viewable_image
from visionentropy.representations import build_representation


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), VisionEntropyApiHandler)
    print(f"VisionEntropy API listening on http://{host}:{port}")
    server.serve_forever()


class VisionEntropyApiHandler(BaseHTTPRequestHandler):
    server_version = "VisionEntropyApi/0.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(api_homepage())
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/datasets":
                self._send_json(dataset_catalog_payload())
                return
            if parsed.path == "/api/synthetic-presets":
                self._send_json(synthetic_presets_payload())
                return
            if parsed.path == "/api/datasets/preview":
                query = parse_query(parsed.query)
                self._send_json(dataset_preview_payload(query))
                return
            if parsed.path == "/api/runs":
                self._send_json(run_history_payload())
                return
            if parsed.path == "/api/comparisons/latest":
                self._send_json(latest_comparison_payload())
                return
            if parsed.path == "/api/results/latest":
                query = parse_query(parsed.query)
                self._send_json(latest_result_payload(query.get("output")))
                return
            if parsed.path == "/api/files":
                query = parse_query(parsed.query)
                self._send_file(query["path"])
                return
            if parsed.path == "/favicon.ico":
                self._send_empty(HTTPStatus.NO_CONTENT)
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as error:  # noqa: BLE001 - local API should report recoverable errors to the UI.
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                payload = self._read_json()
                result = run_vertical_slice(build_run_config(payload))
                self._send_json(run_result_payload(result))
                return
            if parsed.path == "/api/comparisons":
                payload = self._read_json()
                result = run_baseline_entropy_comparison(build_comparison_config(payload))
                self._send_json(comparison_result_payload(result))
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as error:  # noqa: BLE001
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.end_headers()

    def _send_file(self, path_value: str) -> None:
        path = resolve_local_path(path_value)
        if not path.exists() or not path.is_file():
            self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type(path))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def parse_query(query: str) -> dict[str, str]:
    parsed = parse_qs(query)
    return {key: values[-1] for key, values in parsed.items() if values}


def api_homepage() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VisionEntropy API</title>
    <style>
      body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #182026;
        background: #edf2f4;
      }
      main {
        display: grid;
        gap: 18px;
        max-width: 820px;
        margin: 0 auto;
        padding: 44px 24px;
      }
      section {
        border: 1px solid #d8e0e3;
        border-radius: 8px;
        background: #ffffff;
        padding: 18px;
      }
      h1, h2, p { margin-top: 0; }
      h1 { font-size: 28px; }
      h2 { font-size: 16px; }
      a {
        color: #155e6e;
        font-weight: 800;
      }
      code {
        display: inline-block;
        padding: 3px 6px;
        border-radius: 6px;
        background: #edf2f4;
      }
      ul {
        display: grid;
        gap: 8px;
        margin-bottom: 0;
      }
    </style>
  </head>
  <body>
    <main>
      <section>
        <h1>VisionEntropy Local API</h1>
        <p>This service powers the admin dashboard. Open the dashboard here:</p>
        <p><a href="http://127.0.0.1:5173/">http://127.0.0.1:5173/</a></p>
      </section>
      <section>
        <h2>Useful API endpoints</h2>
        <ul>
          <li><code>/api/health</code></li>
          <li><code>/api/datasets</code></li>
          <li><code>/api/datasets/preview?name=synthetic_shapes&amp;sample_index=0&amp;representation=grayscale</code></li>
          <li><code>/api/results/latest</code></li>
        </ul>
      </section>
    </main>
  </body>
</html>
"""


def dataset_catalog_payload() -> dict[str, Any]:
    specs = load_dataset_specs()
    datasets = []
    for spec in specs.values():
        status = dataset_status(spec, project_root=Path.cwd())
        datasets.append(
            {
                "name": status.name,
                "title": status.title,
                "mode": status.mode,
                "ready": status.ready,
                "root": str(status.root) if status.root is not None else None,
                "missingPaths": [str(path) for path in status.missing_paths],
                "message": status.message,
            }
        )
    return {"datasets": datasets}


def synthetic_presets_payload() -> dict[str, Any]:
    labels = {
        "s01_clean_high_contrast": "S01 clean high contrast",
        "s02_gaussian_noise": "S02 Gaussian noise",
        "s03_impulse_noise": "S03 impulse noise",
        "s04_blurred_boundaries": "S04 blurred boundaries",
        "s05_textured_foreground": "S05 textured foreground",
        "s06_textured_background": "S06 textured background",
        "s07_overlapping_objects": "S07 overlapping objects",
        "s08_low_contrast": "S08 low contrast",
    }
    return {
        "presets": [
            {"id": "custom", "label": "Custom", "parameters": {}},
            *[
                {"id": preset, "label": labels[preset], "parameters": parameters}
                for preset, parameters in SYNTHETIC_BENCHMARK_PRESETS.items()
            ],
        ]
    }


def dataset_preview_payload(query: dict[str, str]) -> dict[str, Any]:
    dataset_name = query.get("name", "synthetic_shapes")
    sample_index = int(query.get("sample_index", "0"))
    representation_name = query.get("representation", "grayscale")
    height = int(query.get("height", "256"))
    width = int(query.get("width", "256"))

    sample = _load_sample(
        {
            "name": dataset_name,
            "sample_index": sample_index,
            "image_size": [height, width],
            **synthetic_parameters_from_payload(query),
        }
    )
    sample = _preprocess_sample(
        sample,
        {"resize": {"height": height, "width": width}, "normalization": {"mode": "zero_one"}},
        {"image_size": [height, width]},
    )
    representation = build_representation(representation_name).transform(sample.image)
    directory = Path("outputs/figures/previews") / f"{dataset_name}_{sample_index}_{representation.name}"
    directory.mkdir(parents=True, exist_ok=True)

    original_path = directory / "original.png"
    representation_path = directory / "representation.png"
    imsave(original_path, _to_uint8_rgb(sample.image), check_contrast=False)
    imsave(representation_path, _to_viewable_image(representation.data), check_contrast=False)

    mask_path = None
    if sample.mask is not None:
        mask_path = directory / "mask.png"
        imsave(mask_path, ((np.asarray(sample.mask) > 0).astype(np.uint8) * 255), check_contrast=False)

    return {
        "sampleId": sample.sample_id,
        "dataset": dataset_name,
        "label": sample.label,
        "metadata": sample.metadata,
        "imageShape": list(sample.image.shape),
        "maskShape": list(sample.mask.shape) if sample.mask is not None else None,
        "representation": {
            "name": representation.name,
            "shape": list(representation.data.shape),
            "channels": list(representation.channels),
        },
        "images": {
            "original": file_url(original_path),
            "representation": file_url(representation_path),
            "mask": file_url(mask_path) if mask_path else None,
        },
    }


def build_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_name = payload.get("dataset", "synthetic_shapes")
    sample_index = int(payload.get("sampleIndex", 0))
    height = int(payload.get("height", 256))
    width = int(payload.get("width", 256))
    representation = payload.get("representation", "grayscale")
    entropy_measure = payload.get("entropyMeasure", "shannon")
    entropy_scope = payload.get("entropyScope", "local")
    segmentation_method = payload.get("segmentationMethod", "feature_kmeans")
    bins = int(payload.get("bins", 64))
    window_radius = int(payload.get("windowRadius", 4))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_slug = build_run_slug(
        dataset_name=dataset_name,
        sample_index=sample_index,
        entropy_measure=entropy_measure,
        entropy_scope=entropy_scope,
        segmentation_method=segmentation_method,
        window_radius=window_radius,
        bins=bins,
    )
    output_name = payload.get("outputName") or f"{run_slug}_{timestamp}"

    return {
        "experiment": {
            "name": output_name,
            "output_directory": str(Path("outputs/runs") / output_name),
        },
        "dataset": {
            "name": dataset_name,
            "sample_index": sample_index,
            "image_size": [height, width],
            "sample_count": max(sample_index + 1, 16),
            **synthetic_parameters_from_payload(payload),
        },
        "preprocessing": {
            "resize": {"height": height, "width": width},
            "normalization": {"mode": "zero_one"},
        },
        "representation": {"name": representation},
        "entropy": {
            "name": entropy_measure,
            "scope": entropy_scope,
            "parameters": {"bins": bins, "window_radius": window_radius},
        },
        "segmentation": {
            "name": segmentation_method,
            "parameters": {
                "bins": bins,
                "foreground": payload.get(
                    "foreground",
                    "mask_overlap" if segmentation_method in {"feature_kmeans", "kmeans"} else "high",
                ),
                "random_state": int(payload.get("randomState", 0)),
            },
        },
    }


def build_comparison_config(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_name = payload.get("dataset", "synthetic_shapes")
    sample_index = int(payload.get("sampleIndex", 0))
    height = int(payload.get("height", 256))
    width = int(payload.get("width", 256))
    bins = int(payload.get("bins", 64))
    window_radius = int(payload.get("windowRadius", 4))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_name = payload.get("outputName") or "_".join(
        [
            "comparison",
            slugify(dataset_name.replace("_shapes", "").replace("_examples", "")),
            f"{sample_index:03d}",
            "baseline_vs_entropy",
            f"r{window_radius}",
            f"b{bins}",
            timestamp,
        ]
    )
    return {
        "experiment": {
            "name": output_name,
            "output_directory": str(Path("outputs/runs") / output_name),
        },
        "dataset": {
            "name": dataset_name,
            "sample_index": sample_index,
            "image_size": [height, width],
            "sample_count": max(sample_index + 1, 16),
            **synthetic_parameters_from_payload(payload),
        },
        "preprocessing": {
            "resize": {"height": height, "width": width},
            "normalization": {"mode": "zero_one"},
        },
        "entropy": {
            "name": "shannon",
            "scope": "local",
            "parameters": {"bins": bins, "window_radius": window_radius},
        },
        "comparison": {
            "name": "baseline_vs_entropy",
            "variants": [
                "baseline_a_grayscale_otsu",
                "baseline_b_grayscale_adaptive",
                "experiment_c_local_shannon",
                "experiment_d_grayscale_local_shannon",
                "experiment_e_grayscale_gradient_local_shannon",
            ],
        },
    }


def synthetic_parameters_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    synthetic = payload.get("synthetic", payload)
    preset = synthetic.get("syntheticPreset", synthetic.get("preset", "custom"))
    return {
        "preset": preset,
        "shape_count": int(synthetic.get("shapeCount", synthetic.get("shape_count", 3))),
        "foreground_texture": float(
            synthetic.get("foregroundTexture", synthetic.get("foreground_texture", 0.0))
        ),
        "background_texture": float(
            synthetic.get("backgroundTexture", synthetic.get("background_texture", 0.0))
        ),
        "gaussian_noise": float(synthetic.get("gaussianNoise", synthetic.get("gaussian_noise", 0.0))),
        "impulse_noise": float(synthetic.get("impulseNoise", synthetic.get("impulse_noise", 0.0))),
        "boundary_blur": float(synthetic.get("boundaryBlur", synthetic.get("boundary_blur", 0.7))),
        "illumination_gradient": float(
            synthetic.get("illuminationGradient", synthetic.get("illumination_gradient", 0.0))
        ),
        "allow_overlap": parse_bool(synthetic.get("allowOverlap", synthetic.get("allow_overlap", False))),
        "contrast": float(synthetic.get("syntheticContrast", synthetic.get("contrast", 1.0))),
        "seed": int(synthetic.get("syntheticSeed", synthetic.get("seed", 42))),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_run_slug(
    *,
    dataset_name: str,
    sample_index: int,
    entropy_measure: str,
    entropy_scope: str,
    segmentation_method: str,
    window_radius: int,
    bins: int,
) -> str:
    dataset_slug = dataset_name.replace("_shapes", "").replace("_examples", "")
    segmentation_slug = segmentation_method.replace("maximum_entropy_threshold", "kapur")
    parts = [
        dataset_slug,
        f"{sample_index:03d}",
        entropy_measure,
        entropy_scope,
        segmentation_slug,
        f"r{window_radius}",
        f"b{bins}",
    ]
    return "_".join(slugify(part) for part in parts)


def slugify(value: object) -> str:
    text = str(value).strip().lower()
    cleaned = []
    for character in text:
        cleaned.append(character if character.isalnum() else "_")
    return "_".join("".join(cleaned).split("_"))


def run_result_payload(result: Any) -> dict[str, Any]:
    return {
        "sampleId": result.sample_id,
        "experiment": result.metadata.get("experiment"),
        "algorithm": {
            "entropyMeasure": result.metadata.get("entropy_measure"),
            "entropyScope": result.metadata.get("entropy_scope"),
            "segmentationMethod": result.metadata.get("segmentation_method"),
        },
        "outputDirectory": result.artifacts.get("summary", "").replace("\\", "/").rsplit("/", 1)[0],
        "runMetadata": result.metadata.get("run_metadata"),
        "threshold": result.metadata.get("threshold"),
        "features": {
            "channels": result.metadata.get("feature_channels"),
            "foregroundRule": result.metadata.get("foreground_rule"),
            "foregroundLabel": result.metadata.get("foreground_label"),
            "clusterCenters": result.metadata.get("cluster_centers"),
        },
        "metrics": result.metrics,
        "runtime": result.runtime,
        "artifacts": artifact_urls(result.artifacts),
    }


def comparison_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "artifacts": artifact_urls(result.get("artifacts", {})),
        "variants": [
            {
                **variant,
                "artifacts": artifact_urls(variant.get("artifacts", {})),
            }
            for variant in result.get("variants", [])
        ],
    }


def latest_result_payload(output: str | None = None) -> dict[str, Any]:
    root = resolve_local_path(output) if output else latest_run_directory()
    if root is None:
        return {"ready": False}
    payload = run_directory_payload(root)
    return {"ready": True, **payload}


def latest_comparison_payload() -> dict[str, Any]:
    root = latest_comparison_directory()
    if root is None:
        return {"ready": False}
    comparison_path = root / "comparison.json"
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    return {"ready": True, **comparison_result_payload(payload)}


def run_history_payload() -> dict[str, Any]:
    runs_root = Path("outputs/runs")
    if not runs_root.exists():
        return {"runs": []}

    runs = []
    for directory in runs_root.iterdir():
        if directory.is_dir() and (directory / "metrics.json").exists():
            runs.append(run_directory_payload(directory))
    runs.sort(key=lambda item: item["updatedAt"], reverse=True)
    return {"runs": runs}


def latest_run_directory() -> Path | None:
    runs_root = Path("outputs/runs")
    if not runs_root.exists():
        return None
    candidates = [
        directory
        for directory in runs_root.iterdir()
        if directory.is_dir() and (directory / "metrics.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "metrics.json").stat().st_mtime)


def latest_comparison_directory() -> Path | None:
    runs_root = Path("outputs/runs")
    if not runs_root.exists():
        return None
    candidates = [
        directory
        for directory in runs_root.iterdir()
        if directory.is_dir() and (directory / "comparison.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "comparison.json").stat().st_mtime)


def run_directory_payload(root: Path) -> dict[str, Any]:
    root = resolve_local_path(str(root))
    metrics_path = root / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"Run metrics not found: {root}")

    artifacts = {
        "original_image": str(root / "images/original.png"),
        "representation": str(root / "images/representation.png"),
        "entropy_map": str(root / "images/entropy_map.png"),
        "local_mean": str(root / "images/local_mean.png"),
        "local_variance": str(root / "images/local_variance.png"),
        "gradient_map": str(root / "images/gradient_map.png"),
        "histogram": str(root / "images/histogram.png"),
        "threshold_curve": str(root / "images/threshold_curve.png"),
        "superpixel_map": str(root / "images/superpixel_map.png"),
        "score_map": str(root / "images/score_map.png"),
        "cluster_labels": str(root / "images/cluster_labels.png"),
        "prediction": str(root / "images/prediction.png"),
        "ground_truth": str(root / "images/ground_truth.png"),
        "error_map": str(root / "images/error_map.png"),
        "summary": str(root / "summary.md"),
        "metrics_json": str(metrics_path),
        "run_metadata": str(root / "run_metadata.json"),
    }
    artifacts = {key: value for key, value in artifacts.items() if Path(value).exists()}
    run_metadata = run_metadata_payload(root)
    return {
        "name": root.name,
        "experiment": root.name,
        "outputDirectory": str(root),
        "updatedAt": metrics_path.stat().st_mtime,
        "runMetadata": run_metadata,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "artifacts": artifact_urls(artifacts),
    }


def run_metadata_payload(root: Path) -> dict[str, Any]:
    metadata_path = root / "run_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    config_path = root / "config.yaml"
    runtime_path = root / "runtime.json"
    if not config_path.exists():
        return {"run": root.name}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    runtime = {}
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    return build_run_metadata(config, runtime=runtime)


def artifact_urls(artifacts: dict[str, str | None]) -> dict[str, str | None]:
    return {key: file_url(Path(path)) if path is not None else None for key, path in artifacts.items()}


def file_url(path: Path | None) -> str | None:
    if path is None:
        return None
    return f"/api/files?path={path.as_posix()}"


def resolve_local_path(path_value: str) -> Path:
    raw = Path(unquote(path_value))
    path = raw if raw.is_absolute() else Path.cwd() / raw
    root = Path.cwd().resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("Requested path is outside the workspace")
    return resolved


def content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".json":
        return "application/json"
    if suffix in {".md", ".txt", ".csv", ".yaml", ".yml"}:
        return "text/plain; charset=utf-8"
    if suffix == ".npy":
        return "application/octet-stream"
    return "application/octet-stream"
