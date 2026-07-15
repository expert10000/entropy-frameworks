import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Boxes,
  Braces,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Columns2,
  Database,
  FlaskConical,
  GitBranch,
  HardDrive,
  Image,
  Layers,
  Maximize2,
  Network,
  Palette,
  Play,
  RefreshCw,
  Settings2,
  SlidersHorizontal,
  Table2,
  UploadCloud
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8765";

type DatasetStatus = {
  name: string;
  title: string;
  mode: string;
  ready: boolean;
  root: string | null;
  missingPaths: string[];
  message: string;
};

type PreviewPayload = {
  sampleId: string;
  dataset: string;
  label: string | number | null;
  metadata: Record<string, unknown>;
  imageShape: number[];
  maskShape: number[] | null;
  representation: {
    name: string;
    shape: number[];
    channels: string[];
  };
  images: {
    original: string;
    representation: string;
    mask: string | null;
  };
};

type RunPayload = {
  name?: string;
  sampleId?: string;
  experiment?: string;
  outputDirectory?: string;
  updatedAt?: number;
  runMetadata?: {
    run?: string;
    dataset?: string;
    sample?: number;
    sampleId?: string | null;
    syntheticPreset?: string | null;
    representation?: string;
    entropy?: {
      name?: string;
      scope?: string;
      bins?: number;
      radius?: number;
    };
    segmentation?: {
      name?: string;
      foreground?: string;
    };
    seed?: number;
    runtimeSeconds?: number | null;
  };
  threshold?: number;
  features?: {
    channels?: string[] | null;
    foregroundRule?: string | null;
    foregroundLabel?: number | null;
    clusterCenters?: number[][] | null;
  };
  metrics?: Record<string, number>;
  runtime?: Record<string, number>;
  artifacts?: Record<string, string>;
};

type ComparisonVariant = {
  id: string;
  title: string;
  kind: "baseline" | "entropy";
  description: string;
  threshold: number | null;
  metrics: Record<string, number>;
  artifacts: Record<string, string | null>;
};

type ComparisonPayload = {
  ready?: boolean;
  sampleId?: string;
  experiment?: string;
  outputDirectory?: string;
  runtime?: Record<string, number>;
  parameters?: Record<string, number>;
  artifacts?: Record<string, string | null>;
  variants?: ComparisonVariant[];
  bestVariantId?: string | null;
};

type ApiState = "checking" | "online" | "offline";
type ResultMode = "single" | "compare" | "overlay";

const pipelineStages = [
  "Dataset",
  "Preprocessing",
  "Representation",
  "Entropy",
  "Segmentation",
  "Evaluation",
  "Report"
];

const representationCatalog = [
  {
    name: "RGB",
    shape: "H x W x 3",
    channels: "red, green, blue",
    use: "Baseline image tensor"
  },
  {
    name: "Grayscale",
    shape: "H x W",
    channels: "intensity",
    use: "Region membership feature"
  },
  {
    name: "Local entropy",
    shape: "H x W",
    channels: "boundary uncertainty",
    use: "Boundary and texture feature"
  },
  {
    name: "Gradient",
    shape: "H x W",
    channels: "edge magnitude",
    use: "Boundary strength feature"
  },
  {
    name: "Lab",
    shape: "H x W x 3",
    channels: "l, a, b",
    use: "Color-distance segmentation"
  }
];

const syntheticPresets = [
  {
    id: "custom",
    label: "Custom",
    values: {}
  },
  {
    id: "s01_clean_high_contrast",
    label: "S01 clean high contrast",
    values: { shapeCount: 3, syntheticContrast: 1.15, boundaryBlur: 0.5, illuminationGradient: 0.05, allowOverlap: false, syntheticSeed: 101 }
  },
  {
    id: "s02_gaussian_noise",
    label: "S02 Gaussian noise",
    values: { shapeCount: 3, gaussianNoise: 0.08, syntheticContrast: 1, boundaryBlur: 0.6, allowOverlap: false, syntheticSeed: 102 }
  },
  {
    id: "s03_impulse_noise",
    label: "S03 impulse noise",
    values: { shapeCount: 3, impulseNoise: 0.06, syntheticContrast: 1, boundaryBlur: 0.5, allowOverlap: false, syntheticSeed: 103 }
  },
  {
    id: "s04_blurred_boundaries",
    label: "S04 blurred boundaries",
    values: { shapeCount: 3, boundaryBlur: 2.2, syntheticContrast: 1, allowOverlap: false, syntheticSeed: 104 }
  },
  {
    id: "s05_textured_foreground",
    label: "S05 textured foreground",
    values: { shapeCount: 3, foregroundTexture: 0.35, syntheticContrast: 1, boundaryBlur: 0.6, allowOverlap: false, syntheticSeed: 105 }
  },
  {
    id: "s06_textured_background",
    label: "S06 textured background",
    values: { shapeCount: 3, backgroundTexture: 0.35, syntheticContrast: 1, boundaryBlur: 0.6, allowOverlap: false, syntheticSeed: 106 }
  },
  {
    id: "s07_overlapping_objects",
    label: "S07 overlapping objects",
    values: { shapeCount: 5, syntheticContrast: 1, boundaryBlur: 0.6, allowOverlap: true, syntheticSeed: 107 }
  },
  {
    id: "s08_low_contrast",
    label: "S08 low contrast",
    values: { shapeCount: 3, syntheticContrast: 0.42, boundaryBlur: 0.6, illuminationGradient: 0.08, allowOverlap: false, syntheticSeed: 108 }
  }
];

const artifactOrder = [
  ["original_image", "Original"],
  ["representation", "Representation"],
  ["local_mean", "Local mean"],
  ["local_variance", "Local variance"],
  ["entropy_map", "Entropy map"],
  ["gradient_map", "Gradient"],
  ["histogram", "Histogram"],
  ["threshold_curve", "Threshold curve"],
  ["superpixel_map", "Superpixels"],
  ["score_map", "Score map"],
  ["cluster_labels", "Clusters"],
  ["prediction", "Prediction"],
  ["ground_truth", "Ground truth"],
  ["error_map", "Error map"]
];

function App() {
  const [apiState, setApiState] = useState<ApiState>("checking");
  const [datasets, setDatasets] = useState<DatasetStatus[]>([]);
  const [dataset, setDataset] = useState("synthetic_shapes");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [representation, setRepresentation] = useState("grayscale");
  const [entropyMeasure, setEntropyMeasure] = useState("shannon");
  const [entropyScope, setEntropyScope] = useState("local");
  const [segmentationMethod, setSegmentationMethod] = useState("feature_kmeans");
  const [syntheticPreset, setSyntheticPreset] = useState("s01_clean_high_contrast");
  const [shapeCount, setShapeCount] = useState(3);
  const [foregroundTexture, setForegroundTexture] = useState(0);
  const [backgroundTexture, setBackgroundTexture] = useState(0);
  const [gaussianNoise, setGaussianNoise] = useState(0);
  const [impulseNoise, setImpulseNoise] = useState(0);
  const [boundaryBlur, setBoundaryBlur] = useState(0.5);
  const [illuminationGradient, setIlluminationGradient] = useState(0.05);
  const [allowOverlap, setAllowOverlap] = useState(false);
  const [syntheticContrast, setSyntheticContrast] = useState(1.15);
  const [syntheticSeed, setSyntheticSeed] = useState(101);
  const [height, setHeight] = useState(256);
  const [width, setWidth] = useState(256);
  const [bins, setBins] = useState(64);
  const [windowRadius, setWindowRadius] = useState(4);
  const [preview, setPreview] = useState<PreviewPayload | null>(null);
  const [runResult, setRunResult] = useState<RunPayload | null>(null);
  const [runHistory, setRunHistory] = useState<RunPayload[]>([]);
  const [comparisonResult, setComparisonResult] = useState<ComparisonPayload | null>(null);
  const [statusText, setStatusText] = useState("Starting up");
  const [isRunning, setIsRunning] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [selectedArtifact, setSelectedArtifact] = useState("entropy_map");
  const [resultMode, setResultMode] = useState<ResultMode>("compare");

  const metrics = useMemo(() => {
    const values = runResult?.metrics ?? {};
    return [
      { label: "Mean IoU", value: values.mean_iou, fallback: "0.000" },
      { label: "Dice", value: values.dice, fallback: "0.000" },
      { label: "Pixel accuracy", value: values.pixel_accuracy, fallback: "0.000" },
      { label: "Precision", value: values.precision, fallback: "0.000" }
    ].map((metric) => ({
      label: metric.label,
      value: metric.value == null ? metric.fallback : metric.value.toFixed(3)
    }));
  }, [runResult]);

  useEffect(() => {
    refreshDashboard();
  }, []);

  async function refreshDashboard() {
    setStatusText("Checking local API");
    try {
      await apiFetch("/api/health");
      const datasetPayload = await apiFetch<{ datasets: DatasetStatus[] }>("/api/datasets");
      const latest = await apiFetch<RunPayload & { ready?: boolean }>("/api/results/latest");
      const latestComparison = await apiFetch<ComparisonPayload>("/api/comparisons/latest");
      const history = await apiFetch<{ runs: RunPayload[] }>("/api/runs");
      setDatasets(datasetPayload.datasets);
      if (latest.ready !== false) {
        setRunResult(latest);
      }
      if (latestComparison.ready !== false) {
        setComparisonResult(latestComparison);
      }
      setRunHistory(history.runs);
      setApiState("online");
      setStatusText("Ready");
    } catch {
      setApiState("offline");
      setStatusText("API offline");
    }
  }

  async function loadPreview(index = sampleIndex) {
    setStatusText("Loading dataset sample");
    try {
      const params = new URLSearchParams({
        name: dataset,
        sample_index: String(index),
        representation,
        height: String(height),
        width: String(width),
        ...syntheticQuery()
      });
      const payload = await apiFetch<PreviewPayload>(`/api/datasets/preview?${params.toString()}`);
      setPreview(payload);
      setSampleIndex(index);
      setStatusText(`Loaded ${payload.sampleId}`);
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Preview failed");
    }
  }

  async function stepPreview(delta: number) {
    const nextIndex = Math.max(0, Math.min(sampleLimit - 1, sampleIndex + delta));
    await loadPreview(nextIndex);
  }

  async function runPipeline() {
    setIsRunning(true);
    setStatusText("Running vertical slice");
    try {
      const payload = await apiFetch<RunPayload>("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          sampleIndex,
          representation,
          entropyMeasure,
          entropyScope,
          segmentationMethod,
          height,
          width,
          bins,
          windowRadius,
          synthetic: syntheticPayload()
        })
      });
      setRunResult(payload);
      const history = await apiFetch<{ runs: RunPayload[] }>("/api/runs");
      setRunHistory(history.runs);
      setStatusText("Running baseline comparison");
      setIsComparing(true);
      try {
        const comparison = await requestComparison();
        setComparisonResult(comparison);
        setStatusText(`Run complete: ${payload.experiment}`);
      } catch (error) {
        setStatusText(error instanceof Error ? error.message : "Comparison failed");
      } finally {
        setIsComparing(false);
      }
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Run failed");
    } finally {
      setIsRunning(false);
    }
  }

  async function runComparison() {
    setIsComparing(true);
    setStatusText("Running baseline comparison");
    try {
      const payload = await requestComparison();
      setComparisonResult(payload);
      setStatusText(`Comparison complete: ${payload.experiment}`);
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Comparison failed");
    } finally {
      setIsComparing(false);
    }
  }

  async function requestComparison() {
    return apiFetch<ComparisonPayload>("/api/comparisons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        sampleIndex,
        height,
        width,
        bins,
        windowRadius,
        synthetic: syntheticPayload()
      })
    });
  }

  function syntheticPayload() {
    return {
      syntheticPreset,
      shapeCount,
      foregroundTexture,
      backgroundTexture,
      gaussianNoise,
      impulseNoise,
      boundaryBlur,
      illuminationGradient,
      allowOverlap,
      syntheticContrast,
      syntheticSeed
    };
  }

  function syntheticQuery() {
    return Object.fromEntries(
      Object.entries(syntheticPayload()).map(([key, value]) => [key, String(value)])
    );
  }

  function applySyntheticPreset(presetId: string) {
    setSyntheticPreset(presetId);
    const preset = syntheticPresets.find((item) => item.id === presetId);
    if (!preset) return;
    const values = preset.values as Partial<ReturnType<typeof syntheticPayload>>;
    if (values.shapeCount != null) setShapeCount(values.shapeCount);
    setForegroundTexture(values.foregroundTexture ?? 0);
    setBackgroundTexture(values.backgroundTexture ?? 0);
    setGaussianNoise(values.gaussianNoise ?? 0);
    setImpulseNoise(values.impulseNoise ?? 0);
    if (values.boundaryBlur != null) setBoundaryBlur(values.boundaryBlur);
    setIlluminationGradient(values.illuminationGradient ?? 0);
    if (values.allowOverlap != null) setAllowOverlap(values.allowOverlap);
    if (values.syntheticContrast != null) setSyntheticContrast(values.syntheticContrast);
    if (values.syntheticSeed != null) setSyntheticSeed(values.syntheticSeed);
  }

  const canUseDataset = dataset !== "oxford_iiit_pet";
  const sampleLimit = dataset === "skimage_examples" ? 5 : 16;
  const resultArtifacts = runResult?.artifacts ?? {};
  const comparisonVariants = comparisonResult?.variants ?? [];
  const bestComparison = comparisonVariants.find((variant) => variant.id === comparisonResult?.bestVariantId);
  const selectedArtifactTitle =
    artifactOrder.find(([key]) => key === selectedArtifact)?.[1] ?? "Artifact";
  const runIdPreview = buildRunIdPreview({
    dataset,
    sampleIndex,
    entropyMeasure,
    entropyScope,
    segmentationMethod,
    windowRadius,
    bins
  });

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={22} />
          </div>
          <div>
            <h1>VisionEntropy</h1>
            <span>Admin console</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary">
          <button className="nav-item active" title="Experiments">
            <FlaskConical size={18} />
            <span>Experiments</span>
          </button>
          <button className="nav-item" title="Datasets">
            <Database size={18} />
            <span>Datasets</span>
          </button>
          <button className="nav-item" title="Representations">
            <Boxes size={18} />
            <span>Representations</span>
          </button>
          <button className="nav-item" title="Region graphs">
            <Network size={18} />
            <span>Region graphs</span>
          </button>
          <button className="nav-item" title="Reports">
            <BarChart3 size={18} />
            <span>Reports</span>
          </button>
        </nav>

        <section className="control-panel">
          <div className="panel-heading">
            <Settings2 size={17} />
            <h2>Run Setup</h2>
          </div>

          <label>
            Dataset
            <select
              value={dataset}
              onChange={(event) => {
                setDataset(event.target.value);
                setSampleIndex(0);
                setPreview(null);
              }}
            >
              <option value="synthetic_shapes">synthetic_shapes</option>
              <option value="skimage_examples">skimage_examples</option>
              <option value="oxford_iiit_pet">oxford_iiit_pet</option>
            </select>
          </label>

          <label>
            Sample
            <input
              type="number"
              value={sampleIndex}
              min={0}
              max={sampleLimit - 1}
              onChange={(event) => setSampleIndex(Number(event.target.value))}
            />
          </label>

          {dataset === "synthetic_shapes" && (
            <div className="synthetic-controls">
              <label>
                Benchmark preset
                <select value={syntheticPreset} onChange={(event) => applySyntheticPreset(event.target.value)}>
                  {syntheticPresets.map((preset) => (
                    <option value={preset.id} key={preset.id}>{preset.label}</option>
                  ))}
                </select>
              </label>

              <div className="two-column">
                <label>
                  Shapes
                  <input type="number" value={shapeCount} min={1} max={8} onChange={(event) => setShapeCount(Number(event.target.value))} />
                </label>
                <label>
                  Contrast
                  <input type="number" value={syntheticContrast} min={0.1} max={1.5} step={0.05} onChange={(event) => setSyntheticContrast(Number(event.target.value))} />
                </label>
              </div>

              <div className="two-column">
                <label>
                  Foreground texture
                  <input type="number" value={foregroundTexture} min={0} max={1} step={0.05} onChange={(event) => setForegroundTexture(Number(event.target.value))} />
                </label>
                <label>
                  Background texture
                  <input type="number" value={backgroundTexture} min={0} max={1} step={0.05} onChange={(event) => setBackgroundTexture(Number(event.target.value))} />
                </label>
              </div>

              <div className="two-column">
                <label>
                  Gaussian noise
                  <input type="number" value={gaussianNoise} min={0} max={0.3} step={0.01} onChange={(event) => setGaussianNoise(Number(event.target.value))} />
                </label>
                <label>
                  Impulse noise
                  <input type="number" value={impulseNoise} min={0} max={0.3} step={0.01} onChange={(event) => setImpulseNoise(Number(event.target.value))} />
                </label>
              </div>

              <div className="two-column">
                <label>
                  Boundary blur
                  <input type="number" value={boundaryBlur} min={0} max={4} step={0.1} onChange={(event) => setBoundaryBlur(Number(event.target.value))} />
                </label>
                <label>
                  Illumination
                  <input type="number" value={illuminationGradient} min={0} max={1} step={0.05} onChange={(event) => setIlluminationGradient(Number(event.target.value))} />
                </label>
              </div>

              <div className="two-column">
                <label>
                  Seed
                  <input type="number" value={syntheticSeed} min={0} max={9999} onChange={(event) => setSyntheticSeed(Number(event.target.value))} />
                </label>
                <label className="checkbox-label">
                  <input type="checkbox" checked={allowOverlap} onChange={(event) => setAllowOverlap(event.target.checked)} />
                  <span>Overlap</span>
                </label>
              </div>
            </div>
          )}

          <label>
            Representation
            <select value={representation} onChange={(event) => setRepresentation(event.target.value)}>
              <option value="rgb">rgb</option>
              <option value="grayscale">grayscale</option>
              <option value="lab">lab</option>
              <option value="red">red</option>
              <option value="green">green</option>
              <option value="blue">blue</option>
            </select>
          </label>

          <label>
            Entropy measure
            <select value={entropyMeasure} onChange={(event) => setEntropyMeasure(event.target.value)}>
              <option value="shannon">Shannon</option>
              <option value="renyi" disabled>Renyi pending</option>
              <option value="tsallis" disabled>Tsallis pending</option>
            </select>
          </label>

          <label>
            Entropy scope
            <select value={entropyScope} onChange={(event) => setEntropyScope(event.target.value)}>
              <option value="local">local</option>
              <option value="global" disabled>global pending</option>
              <option value="region" disabled>region pending</option>
            </select>
          </label>

          <label>
            Segmentation method
            <select value={segmentationMethod} onChange={(event) => setSegmentationMethod(event.target.value)}>
              <option value="feature_kmeans">Feature stack KMeans</option>
              <option value="kapur">Kapur maximum entropy</option>
              <option value="otsu" disabled>Otsu threshold pending</option>
              <option value="local_adaptive" disabled>Local adaptive pending</option>
              <option value="entropy_intensity" disabled>Entropy + intensity pending</option>
            </select>
          </label>

          <div className="two-column">
            <label>
              Height
              <input type="number" value={height} min={32} max={768} onChange={(event) => setHeight(Number(event.target.value))} />
            </label>
            <label>
              Width
              <input type="number" value={width} min={32} max={768} onChange={(event) => setWidth(Number(event.target.value))} />
            </label>
          </div>

          <div className="two-column">
            <label>
              Bins
              <input type="number" value={bins} min={2} max={512} onChange={(event) => setBins(Number(event.target.value))} />
            </label>
            <label>
              Radius
              <input
                type="number"
                value={windowRadius}
                min={1}
                max={16}
                onChange={(event) => setWindowRadius(Number(event.target.value))}
              />
            </label>
          </div>

          <button className="secondary-action" onClick={() => loadPreview()} disabled={!canUseDataset || apiState !== "online"}>
            <Image size={18} />
            <span>Load Sample</span>
          </button>

          <button
            className="primary-action"
            onClick={runPipeline}
            disabled={!canUseDataset || isRunning || isComparing || apiState !== "online"}
          >
            <Play size={18} />
            <span>{isRunning ? "Running" : "Run Slice"}</span>
          </button>

          <button
            className="secondary-action"
            onClick={runComparison}
            disabled={!canUseDataset || isRunning || isComparing || apiState !== "online"}
          >
            <BarChart3 size={18} />
            <span>{isComparing ? "Comparing" : "Run Comparison"}</span>
          </button>

          <div className="run-id-preview">
            <span>Run ID</span>
            <code>{runIdPreview}</code>
          </div>
        </section>

        <section className="control-panel">
          <div className="panel-heading">
            <HardDrive size={17} />
            <h2>Local API</h2>
          </div>
          <div className="api-status">
            <span className={apiState === "online" ? "stage-dot complete" : "stage-dot"} />
            <strong>{statusText}</strong>
          </div>
          <button className="secondary-action" onClick={refreshDashboard}>
            <RefreshCw size={17} />
            <span>Refresh</span>
          </button>
          <button className="secondary-action">
            <UploadCloud size={17} />
            <span>Attach Dataset</span>
          </button>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">VE-0.1 Classical Vertical Slice</p>
            <h2>Entropy-Guided Image Segmentation</h2>
          </div>
          <span className={apiState === "online" ? "status-pill ready" : "status-pill missing"}>
            {apiState === "online" ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
            {apiState === "online" ? "API online" : "API offline"}
          </span>
        </header>

        <section className="status-strip">
          {pipelineStages.map((stage, index) => (
            <div className="stage" key={stage}>
              <span className={runResult || index < 3 ? "stage-dot complete" : "stage-dot"} />
              <span>{stage}</span>
            </div>
          ))}
        </section>

        <section className="metrics-grid">
          {metrics.map((metric) => (
            <article className="metric-card" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
              <em>{runResult?.experiment ?? "no run"}</em>
            </article>
          ))}
        </section>

        <section className="surface dataset-library">
          <div className="surface-heading split">
            <div className="surface-title">
              <Database size={18} />
              <h3>Dataset Library</h3>
            </div>
            <button className="secondary-action compact" onClick={refreshDashboard}>
              <RefreshCw size={17} />
              <span>Check Status</span>
            </button>
          </div>

          <div className="dataset-grid">
            {datasets.map((item) => (
              <article className="dataset-card" key={item.name}>
                <div className="dataset-card-heading">
                  <div>
                    <h4>{item.title}</h4>
                    <span>{item.name}</span>
                  </div>
                  <span className={item.ready ? "status-pill ready" : "status-pill missing"}>
                    {item.ready ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                    {item.ready ? "Ready" : "Needs files"}
                  </span>
                </div>
                <dl>
                  <div>
                    <dt>Mode</dt>
                    <dd>{item.mode}</dd>
                  </div>
                  <div>
                    <dt>Root</dt>
                    <dd>{item.root ?? "none"}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        </section>

        <section className="surface dataset-viewer">
          <div className="surface-heading split">
            <div className="surface-title">
              <Image size={18} />
              <h3>Dataset Viewer</h3>
            </div>
            <div className="viewer-controls">
              <button
                className="icon-action"
                title="Previous sample"
                onClick={() => stepPreview(-1)}
                disabled={!canUseDataset || apiState !== "online" || sampleIndex <= 0}
              >
                <ChevronLeft size={17} />
              </button>
              <span className={preview ? "status-pill ready" : "status-pill missing"}>
                {preview ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                {preview ? preview.sampleId : "No sample loaded"}
              </span>
              <button
                className="icon-action"
                title="Next sample"
                onClick={() => stepPreview(1)}
                disabled={!canUseDataset || apiState !== "online" || sampleIndex >= sampleLimit - 1}
              >
                <ChevronRight size={17} />
              </button>
            </div>
          </div>
          <div className="live-viewer-grid">
            <ImagePanel title="Original" src={preview?.images.original} />
            <ImagePanel title={`Representation ${preview?.representation.name ?? ""}`} src={preview?.images.representation} />
            <ImagePanel title="Mask" src={preview?.images.mask} />
          </div>
          {preview && (
            <div className="viewer-details">
              <dl className="parameter-list compact-list">
                <div>
                  <dt>Image</dt>
                  <dd>{preview.imageShape.join(" x ")}</dd>
                </div>
                <div>
                  <dt>Representation</dt>
                  <dd>{preview.representation.shape.join(" x ")}</dd>
                </div>
                <div>
                  <dt>Channels</dt>
                  <dd>{preview.representation.channels.join(", ")}</dd>
                </div>
                <div>
                  <dt>Label</dt>
                  <dd>{preview.label ?? "none"}</dd>
                </div>
              </dl>
              <pre className="metadata-preview">{JSON.stringify(preview.metadata, null, 2)}</pre>
            </div>
          )}
        </section>

        <section className="prep-grid">
          <article className="surface">
            <div className="surface-heading">
              <Maximize2 size={18} />
              <h3>Preprocessing</h3>
            </div>
            <dl className="parameter-list">
              <div>
                <dt>Resize</dt>
                <dd>{height} x {width}</dd>
              </div>
              <div>
                <dt>Normalize</dt>
                <dd>zero_one</dd>
              </div>
              <div>
                <dt>Entropy bins</dt>
                <dd>{bins}</dd>
              </div>
              <div>
                <dt>Window radius</dt>
                <dd>{windowRadius}</dd>
              </div>
              <div>
                <dt>Entropy</dt>
                <dd>{entropyMeasure} / {entropyScope}</dd>
              </div>
              <div>
                <dt>Segmenter</dt>
                <dd>{segmentationMethod === "feature_kmeans" ? "feature KMeans" : segmentationMethod}</dd>
              </div>
              <div>
                <dt>Preset</dt>
                <dd>{dataset === "synthetic_shapes" ? syntheticPreset.replace(/_/g, " ") : "none"}</dd>
              </div>
              <div>
                <dt>Feature stack</dt>
                <dd>{segmentationMethod === "feature_kmeans" ? "I + entropy + gradient" : "entropy map"}</dd>
              </div>
            </dl>
          </article>

          <article className="surface representation-library">
            <div className="surface-heading">
              <Palette size={18} />
              <h3>Representations</h3>
            </div>
            <div className="representation-grid">
              {representationCatalog.map((item) => (
                <article className="representation-card" key={item.name}>
                  <h4>{item.name}</h4>
                  <span>{item.shape}</span>
                  <p>{item.channels}</p>
                  <em>{item.use}</em>
                </article>
              ))}
            </div>
          </article>
        </section>

        <section className="surface run-history">
          <div className="surface-heading split">
            <div className="surface-title">
              <Table2 size={18} />
              <h3>Run History</h3>
            </div>
            <span className="status-pill ready">{runHistory.length} runs</span>
          </div>
          <div className="run-history-grid">
            {runHistory.map((run) => (
              <button
                className={(runResult?.experiment ?? runResult?.name) === (run.experiment ?? run.name) ? "run-card active" : "run-card"}
                key={run.name ?? run.outputDirectory}
                onClick={() => {
                  setRunResult(run);
                  setStatusText(`Loaded run: ${run.experiment ?? run.name}`);
                }}
              >
                <strong>{run.experiment ?? run.name}</strong>
                <span>{formatRunSubtitle(run)}</span>
                <dl className="run-metadata-list">
                  <div>
                    <dt>Dataset</dt>
                    <dd>{run.runMetadata?.dataset ?? "unknown"}</dd>
                  </div>
                  <div>
                    <dt>Sample</dt>
                    <dd>{run.runMetadata?.sample ?? run.sampleId ?? "unknown"}</dd>
                  </div>
                  <div>
                    <dt>Representation</dt>
                    <dd>{run.runMetadata?.representation ?? "unknown"}</dd>
                  </div>
                  <div>
                    <dt>Entropy</dt>
                    <dd>{formatEntropy(run)}</dd>
                  </div>
                  <div>
                    <dt>Segmentation</dt>
                    <dd>{formatSegmentation(run)}</dd>
                  </div>
                  <div>
                    <dt>Bins</dt>
                    <dd>{run.runMetadata?.entropy?.bins ?? "unknown"}</dd>
                  </div>
                  <div>
                    <dt>Radius</dt>
                    <dd>{run.runMetadata?.entropy?.radius ?? "unknown"}</dd>
                  </div>
                  <div>
                    <dt>Seed</dt>
                    <dd>{run.runMetadata?.seed ?? "unknown"}</dd>
                  </div>
                  <div>
                    <dt>Runtime</dt>
                    <dd>{formatRuntime(run.runMetadata?.runtimeSeconds ?? run.runtime?.duration_seconds)}</dd>
                  </div>
                </dl>
                <dl className="run-score-list">
                  <div>
                    <dt>IoU</dt>
                    <dd>{formatMetric(run.metrics?.mean_iou)}</dd>
                  </div>
                  <div>
                    <dt>Dice</dt>
                    <dd>{formatMetric(run.metrics?.dice)}</dd>
                  </div>
                  <div>
                    <dt>Accuracy</dt>
                    <dd>{formatMetric(run.metrics?.pixel_accuracy)}</dd>
                  </div>
                </dl>
              </button>
            ))}
          </div>
        </section>

        <section className="surface baseline-comparison">
          <div className="surface-heading split">
            <div className="surface-title">
              <BarChart3 size={18} />
              <h3>Baseline Comparison</h3>
            </div>
            <div className="comparison-actions">
              {bestComparison && <span className="status-pill ready">Best: {bestComparison.title.replace(/^Experiment |^Baseline /, "")}</span>}
              <button
                className="secondary-action compact"
                onClick={runComparison}
                disabled={!canUseDataset || isRunning || isComparing || apiState !== "online"}
              >
                <Play size={16} />
                <span>{isComparing ? "Comparing" : "Run"}</span>
              </button>
            </div>
          </div>

          <div className="comparison-grid">
            {comparisonVariants.length > 0 ? (
              comparisonVariants.map((variant) => (
                <ComparisonVariantCard
                  key={variant.id}
                  variant={variant}
                  isBest={variant.id === comparisonResult?.bestVariantId}
                />
              ))
            ) : (
              <div className="empty-comparison">
                <BarChart3 size={22} />
                <span>No comparison run yet</span>
              </div>
            )}
          </div>
        </section>

        <section className="content-grid">
          <article className="surface large">
            <div className="surface-heading split">
              <div className="surface-title">
                <Image size={18} />
                <h3>Result Viewer</h3>
              </div>
              <div className="segmented-control">
                <button
                  className={resultMode === "single" ? "active" : ""}
                  onClick={() => setResultMode("single")}
                  title="Single artifact"
                >
                  <Image size={16} />
                  <span>Single</span>
                </button>
                <button
                  className={resultMode === "compare" ? "active" : ""}
                  onClick={() => setResultMode("compare")}
                  title="Compare with original"
                >
                  <Columns2 size={16} />
                  <span>Compare</span>
                </button>
                <button
                  className={resultMode === "overlay" ? "active" : ""}
                  onClick={() => setResultMode("overlay")}
                  title="Overlay prediction"
                >
                  <Layers size={16} />
                  <span>Overlay</span>
                </button>
              </div>
            </div>
            <div className="artifact-tabs">
              {artifactOrder.map(([key, title]) => (
                <button
                  key={key}
                  className={selectedArtifact === key ? "active" : ""}
                  onClick={() => setSelectedArtifact(key)}
                  disabled={!resultArtifacts[key]}
                >
                  {title}
                </button>
              ))}
            </div>
            <ResultArtifactViewer
              mode={resultMode}
              selectedTitle={selectedArtifactTitle}
              original={resultArtifacts.original_image}
              selected={resultArtifacts[selectedArtifact]}
              prediction={resultArtifacts.prediction}
            />
          </article>

          <article className="surface">
            <div className="surface-heading">
              <GitBranch size={18} />
              <h3>Segmentation</h3>
            </div>
            <div className="mini-artifacts">
              <ImagePanel title="Ground truth" src={resultArtifacts.ground_truth} />
              <ImagePanel title="Error map" src={resultArtifacts.error_map} />
            </div>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <Table2 size={18} />
              <h3>Artifacts</h3>
            </div>
            <ul className="artifact-list">
              {artifactOrder.map(([key, title]) => (
                <li
                  key={key}
                  className={selectedArtifact === key ? "active" : ""}
                  onClick={() => {
                    if (resultArtifacts[key]) setSelectedArtifact(key);
                  }}
                >
                  {resultArtifacts[key] ? title : `${title} pending`}
                </li>
              ))}
            </ul>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <SlidersHorizontal size={18} />
              <h3>Parameters</h3>
            </div>
            <dl className="parameter-list">
              <div>
                <dt>Dataset</dt>
                <dd>{dataset}</dd>
              </div>
              <div>
                <dt>Sample</dt>
                <dd>{sampleIndex}</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>{runResult?.experiment ?? "none"}</dd>
              </div>
              <div>
                <dt>Foreground rule</dt>
                <dd>{runResult?.features?.foregroundRule ?? "none"}</dd>
              </div>
            </dl>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <Braces size={18} />
              <h3>Run Config</h3>
            </div>
            <pre>{`dataset: ${dataset}
representation: ${representation}
synthetic:
  preset: ${dataset === "synthetic_shapes" ? syntheticPreset : "none"}
  shapes: ${shapeCount}
  gaussian_noise: ${gaussianNoise}
  impulse_noise: ${impulseNoise}
  boundary_blur: ${boundaryBlur}
  contrast: ${syntheticContrast}
entropy:
  name: ${entropyMeasure}
  scope: ${entropyScope}
  bins: ${bins}
  window_radius: ${windowRadius}
segmentation:
  name: ${segmentationMethod}
  foreground: ${segmentationMethod === "feature_kmeans" ? "mask_overlap_eval" : "high"}
features:
  ${segmentationMethod === "feature_kmeans" ? "[grayscale, local_entropy, gradient_magnitude]" : "entropy_map"}
run_id:
  ${runIdPreview}_<timestamp>`}</pre>
          </article>
        </section>
      </section>
    </main>
  );
}

function ImagePanel({ title, src }: { title: string; src?: string | null }) {
  return (
    <figure className="image-panel">
      {src ? <img src={apiFileUrl(src)} alt={title} /> : <div className="empty-image" />}
      <figcaption>{title}</figcaption>
    </figure>
  );
}

function ResultArtifactViewer({
  mode,
  selectedTitle,
  original,
  selected,
  prediction
}: {
  mode: ResultMode;
  selectedTitle: string;
  original?: string;
  selected?: string;
  prediction?: string;
}) {
  if (mode === "single") {
    return (
      <div className="artifact-board single">
        <ImagePanel title={selectedTitle} src={selected} />
      </div>
    );
  }

  if (mode === "overlay") {
    return (
      <figure className="overlay-panel">
        {original ? <img src={apiFileUrl(original)} alt="Original" /> : <div className="empty-image" />}
        {prediction && <img className="overlay-image" src={apiFileUrl(prediction)} alt="Prediction overlay" />}
        <figcaption>Prediction overlay</figcaption>
      </figure>
    );
  }

  return (
    <div className="artifact-board">
      <ImagePanel title="Original" src={original} />
      <ImagePanel title={selectedTitle} src={selected} />
    </div>
  );
}

function ComparisonVariantCard({ variant, isBest }: { variant: ComparisonVariant; isBest: boolean }) {
  return (
    <article className={isBest ? "comparison-card best" : "comparison-card"}>
      <div className="comparison-card-heading">
        <div>
          <strong>{variant.title}</strong>
          <span>{variant.kind}</span>
        </div>
        {isBest && <span className="status-pill ready">Best</span>}
      </div>
      <p>{variant.description}</p>
      <dl>
        <div>
          <dt>IoU</dt>
          <dd>{formatMetric(variant.metrics.mean_iou)}</dd>
        </div>
        <div>
          <dt>Dice</dt>
          <dd>{formatMetric(variant.metrics.dice)}</dd>
        </div>
        <div>
          <dt>Accuracy</dt>
          <dd>{formatMetric(variant.metrics.pixel_accuracy)}</dd>
        </div>
      </dl>
      <div className="comparison-images">
        <ImagePanel title="Score" src={variant.artifacts.score_map} />
        <ImagePanel title="Prediction" src={variant.artifacts.prediction} />
        <ImagePanel title="Error" src={variant.artifacts.error_map} />
      </div>
    </article>
  );
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? "API request failed");
  }
  return payload as T;
}

function apiFileUrl(path: string) {
  return path.startsWith("/api/") ? `${API_BASE}${path}` : path;
}

function formatMetric(value?: number) {
  return value == null ? "0.000" : value.toFixed(3);
}

function formatRuntime(value?: number | null) {
  return value == null ? "unknown" : `${value.toFixed(2)} s`;
}

function formatEntropy(run: RunPayload) {
  const entropy = run.runMetadata?.entropy;
  if (!entropy) return "unknown";
  return `${entropy.scope ?? "local"} ${entropy.name ?? "shannon"}`;
}

function formatSegmentation(run: RunPayload) {
  const name = run.runMetadata?.segmentation?.name;
  if (!name) return "unknown";
  if (name === "feature_kmeans") return "intensity + entropy k-means";
  if (name === "kapur" || name === "maximum_entropy_threshold") return "Kapur entropy threshold";
  return name.replace(/_/g, " ");
}

function formatRunSubtitle(run: RunPayload) {
  const metadata = run.runMetadata;
  if (!metadata) return run.outputDirectory ?? "outputs/runs";
  const preset = metadata.syntheticPreset && metadata.syntheticPreset !== "custom"
    ? ` / ${metadata.syntheticPreset.replace(/_/g, " ")}`
    : "";
  return `${metadata.dataset ?? "dataset"} / sample ${metadata.sample ?? "?"}${preset}`;
}

function buildRunIdPreview({
  dataset,
  sampleIndex,
  entropyMeasure,
  entropyScope,
  segmentationMethod,
  windowRadius,
  bins
}: {
  dataset: string;
  sampleIndex: number;
  entropyMeasure: string;
  entropyScope: string;
  segmentationMethod: string;
  windowRadius: number;
  bins: number;
}) {
  const datasetSlug = dataset.replace("_shapes", "").replace("_examples", "");
  return [
    datasetSlug,
    String(sampleIndex).padStart(3, "0"),
    entropyMeasure,
    entropyScope,
    segmentationMethod,
    `r${windowRadius}`,
    `b${bins}`
  ]
    .map((part) => part.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""))
    .join("_");
}

export default App;
