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
  threshold?: number;
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
    use: "Local entropy and thresholding"
  },
  {
    name: "Lab",
    shape: "H x W x 3",
    channels: "l, a, b",
    use: "Color-distance segmentation"
  }
];

const artifactOrder = [
  ["original_image", "Original"],
  ["representation", "Representation"],
  ["entropy_map", "Entropy map"],
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
  const [segmentationMethod, setSegmentationMethod] = useState("kapur");
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
        width: String(width)
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
          windowRadius
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
        windowRadius
      })
    });
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
              <option value="kapur">Kapur maximum entropy</option>
              <option value="otsu" disabled>Otsu threshold pending</option>
              <option value="local_adaptive" disabled>Local adaptive pending</option>
              <option value="kmeans" disabled>k-means pending</option>
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
                <dd>{segmentationMethod}</dd>
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
                <span>{run.outputDirectory ?? "outputs/runs"}</span>
                <dl>
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
            </dl>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <Braces size={18} />
              <h3>Run Config</h3>
            </div>
            <pre>{`dataset: ${dataset}
representation: ${representation}
entropy:
  name: ${entropyMeasure}
  scope: ${entropyScope}
  bins: ${bins}
  window_radius: ${windowRadius}
segmentation:
  name: ${segmentationMethod}
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
