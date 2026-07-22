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
  XCircle,
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
  regions?: {
    count?: number;
    edge_count?: number;
  };
  graph?: {
    mean_node_entropy?: number;
    mean_edge_entropy?: number;
    spectral_entropy?: number;
    normalized_spectral_entropy?: number;
    partition_count?: number;
  } | null;
  deep?: {
    available?: boolean;
    model?: string;
    layer?: string;
    feature_shape?: number[];
    latent_size?: number;
    class_count?: number;
    top_probabilities?: Array<{ index: number; probability: number }>;
    mean_activation_entropy?: number;
    mean_fuzzy_entropy?: number;
    mean_rough_uncertainty?: number;
    mean_fuzzy_rough_uncertainty?: number;
    latent_entropy?: number;
    predictive_entropy?: number;
    representation_level?: string;
    uncertainty_method?: string;
    neighborhood_k?: number;
    similarity_sigma?: number;
  } | null;
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
type WorkspaceView = "experiments" | "deep";
type ActiveRunKind = "slice" | "comparison" | "deep";
type ActiveRunState = {
  kind: ActiveRunKind;
  startedAt: number;
};
type JobStatusPayload = {
  jobId: string;
  kind: ActiveRunKind;
  state: "queued" | "running" | "cancelling" | "cancelled" | "complete" | "error";
  stage: string;
  stageIndex: number;
  totalStages: number;
  percent: number;
  startedAt: number;
  updatedAt: number;
  finishedAt?: number | null;
  elapsedSeconds: number;
  result?: RunPayload | ComparisonPayload | null;
  error?: string | null;
  cancelRequested?: boolean;
  timeline?: Array<{
    stage: string;
    stageIndex: number;
    percent: number;
    startedAt: number;
    finishedAt?: number | null;
    durationSeconds: number;
  }>;
};
type RunProgressModel = {
  label: string;
  percent: number;
  elapsedSeconds: number;
  currentStage: string;
  stages: string[];
  activeStageIndex: number;
};

const pipelineStages = [
  "Dataset",
  "Preprocessing",
  "Representation",
  "Entropy",
  "Segmentation",
  "Evaluation",
  "Report"
];

const deepPipelineStages = [
  "Image",
  "CNN / ResNet",
  "Selected layer",
  "Feature representation",
  "Uncertainty representation",
  "Entropy calculation",
  "Evaluation"
];

const runProgressProfiles: Record<ActiveRunKind, { label: string; expectedSeconds: number; stages: string[] }> = {
  slice: {
    label: "Running slice",
    expectedSeconds: 10,
    stages: ["Dataset", "Preprocess", "Features", "Entropy", "Segmentation", "Deep outputs", "Artifacts"]
  },
  comparison: {
    label: "Running comparison",
    expectedSeconds: 7,
    stages: ["Baseline A", "Baseline B", "Entropy variants", "Metrics", "Report"]
  },
  deep: {
    label: "Running deep entropy",
    expectedSeconds: 12,
    stages: ["Image", "CNN / ResNet", "Layer capture", "Fuzzy entropy", "Rough uncertainty", "Artifacts"]
  }
};

const deepModelOptions = [
  { value: "small_cnn", label: "Small CNN", layers: ["stem", "layer1", "layer2", "avgpool", "logits"] },
  { value: "resnet18", label: "ResNet-18", layers: ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool", "logits"] },
  { value: "resnet34", label: "ResNet-34", layers: ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool", "logits"] }
];

const featureRepresentationOptions = [
  { value: "pixel_embedding", label: "Pixel embedding", basis: "z_xy from C x H x W feature maps" },
  { value: "patch_embedding", label: "Patch embedding", basis: "local neighborhoods over feature cells" },
  { value: "superpixel_embedding", label: "Superpixel embedding", basis: "region-pooled CNN descriptors" },
  { value: "image_embedding", label: "Image embedding", basis: "avgpool latent vector" }
];

const uncertaintyMethodOptions = [
  { value: "classical", label: "Classical", output: "activation, latent, predictive entropy" },
  { value: "fuzzy", label: "Fuzzy", output: "membership entropy over feature similarity" },
  { value: "rough", label: "Rough", output: "lower / upper approximation uncertainty" },
  { value: "fuzzy_rough", label: "Fuzzy-rough", output: "rough neighborhoods weighted by fuzzy relation" },
  { value: "predictive", label: "Predictive", output: "softmax distribution entropy" }
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
  ["region_labels", "Region labels"],
  ["region_mean", "Region mean"],
  ["region_entropy", "Region entropy"],
  ["region_graph", "Region graph"],
  ["graph_node_entropy", "Node entropy"],
  ["graph_edge_entropy", "Edge entropy"],
  ["graph_spectral_entropy", "Spectral entropy"],
  ["graph_partition", "Graph partition"],
  ["deep_feature_map", "Deep feature map"],
  ["activation_entropy", "Activation entropy"],
  ["fuzzy_entropy", "Fuzzy entropy"],
  ["rough_uncertainty", "Rough uncertainty"],
  ["fuzzy_rough_uncertainty", "Fuzzy-rough uncertainty"],
  ["latent_entropy", "Latent entropy"],
  ["predictive_entropy", "Predictive entropy"],
  ["score_map", "Score map"],
  ["cluster_labels", "Clusters"],
  ["prediction", "Prediction"],
  ["ground_truth", "Ground truth"],
  ["error_map", "Error map"]
];

const segmentationOptions = [
  { value: "otsu", label: "Otsu threshold" },
  { value: "kapur", label: "Kapur maximum entropy" },
  { value: "local_adaptive", label: "Adaptive threshold" },
  { value: "feature_kmeans", label: "Feature stack KMeans" },
  { value: "gaussian_mixture", label: "Gaussian mixture" },
  { value: "random_forest", label: "Random Forest" },
  { value: "watershed", label: "Watershed" },
  { value: "region_growing", label: "Region growing" }
];

const featureStackMethods = new Set(["feature_kmeans", "gaussian_mixture", "random_forest"]);
const maskOverlapMethods = new Set(["feature_kmeans", "gaussian_mixture"]);
const regionMethods = new Set(["watershed", "region_growing"]);

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
  const [activeRun, setActiveRun] = useState<ActiveRunState | null>(null);
  const [activeJob, setActiveJob] = useState<JobStatusPayload | null>(null);
  const [lastJob, setLastJob] = useState<JobStatusPayload | null>(null);
  const [isJobDetailsOpen, setIsJobDetailsOpen] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [activeView, setActiveView] = useState<WorkspaceView>("experiments");
  const [deepModel, setDeepModel] = useState("resnet18");
  const [deepLayer, setDeepLayer] = useState("layer4");
  const [deepRepresentationLevel, setDeepRepresentationLevel] = useState("pixel_embedding");
  const [deepUncertaintyMethod, setDeepUncertaintyMethod] = useState("classical");
  const [deepImageSize, setDeepImageSize] = useState(128);
  const [deepNeighborhoodK, setDeepNeighborhoodK] = useState(15);
  const [deepSimilaritySigma, setDeepSimilaritySigma] = useState(0);
  const [selectedArtifact, setSelectedArtifact] = useState("entropy_map");
  const [resultMode, setResultMode] = useState<ResultMode>("compare");

  const metrics = useMemo(() => {
    const values = runResult?.metrics ?? {};
    return [
      { label: "Mean IoU", value: values.mean_iou, fallback: "0.000" },
      { label: "Dice", value: values.dice, fallback: "0.000" },
      { label: "Boundary F1", value: values.boundary_f1, fallback: "0.000" },
      { label: "Error AUROC", value: values.error_detection_auroc, fallback: "0.000" }
    ].map((metric) => ({
      label: metric.label,
      value: metric.value == null ? metric.fallback : metric.value.toFixed(3)
    }));
  }, [runResult]);

  const secondaryMetrics = useMemo(() => {
    const values = runResult?.metrics ?? {};
    return [
      { label: "Pixel accuracy", value: values.pixel_accuracy },
      { label: "Precision", value: values.precision },
      { label: "Recall", value: values.recall },
      { label: "Specificity", value: values.specificity }
    ].map((metric) => ({
      label: metric.label,
      value: metric.value == null ? "0.000" : metric.value.toFixed(3)
    }));
  }, [runResult]);

  const confusion = useMemo(() => {
    const values = runResult?.metrics ?? {};
    return {
      trueNegative: values.true_negative ?? 0,
      falsePositive: values.false_positive ?? 0,
      falseNegative: values.false_negative ?? 0,
      truePositive: values.true_positive ?? 0
    };
  }, [runResult]);
  const activeProgress = useMemo(
    () => (activeJob ? buildJobProgress(activeJob) : activeRun ? buildRunProgress(activeRun, elapsedSeconds) : null),
    [activeJob, activeRun, elapsedSeconds]
  );

  useEffect(() => {
    refreshDashboard();
  }, []);

  useEffect(() => {
    if (!activeRun) {
      setElapsedSeconds(0);
      return;
    }
    const updateElapsed = () => setElapsedSeconds((Date.now() - activeRun.startedAt) / 1000);
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 250);
    return () => window.clearInterval(interval);
  }, [activeRun]);

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
    setActiveRun({ kind: "slice", startedAt: Date.now() });
    setIsJobDetailsOpen(false);
    setStatusText("Running vertical slice");
    try {
      const payload = await requestRunJob("slice");
      setRunResult(payload);
      const history = await apiFetch<{ runs: RunPayload[] }>("/api/runs");
      setRunHistory(history.runs);
      setStatusText("Running baseline comparison");
      setIsComparing(true);
      setActiveRun({ kind: "comparison", startedAt: Date.now() });
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
      setActiveRun(null);
      setActiveJob(null);
    }
  }

  async function runComparison() {
    setIsComparing(true);
    setActiveRun({ kind: "comparison", startedAt: Date.now() });
    setIsJobDetailsOpen(false);
    setStatusText("Running baseline comparison");
    try {
      const payload = await requestComparison();
      setComparisonResult(payload);
      setStatusText(`Comparison complete: ${payload.experiment}`);
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Comparison failed");
    } finally {
      setIsComparing(false);
      setActiveRun(null);
      setActiveJob(null);
    }
  }

  async function requestComparison() {
    const job = await apiFetch<JobStatusPayload>("/api/jobs/comparisons", {
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
    return pollJob<ComparisonPayload>(job);
  }

  async function runDeepPipeline() {
    setIsRunning(true);
    setActiveRun({ kind: "deep", startedAt: Date.now() });
    setIsJobDetailsOpen(false);
    setStatusText("Running deep entropy slice");
    try {
      const payload = await requestRunJob("deep");
      setRunResult(payload);
      const history = await apiFetch<{ runs: RunPayload[] }>("/api/runs");
      setRunHistory(history.runs);
      setSelectedArtifact(payload.artifacts?.activation_entropy ? "activation_entropy" : "deep_feature_map");
      setStatusText(`Deep entropy run complete: ${payload.experiment}`);
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Deep entropy run failed");
    } finally {
      setIsRunning(false);
      setActiveRun(null);
      setActiveJob(null);
    }
  }

  async function requestRunJob(jobKind: "slice" | "deep") {
    const job = await apiFetch<JobStatusPayload>("/api/jobs/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jobKind,
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
        deepEnabled: true,
        deepModel,
        deepLayer,
        deepRepresentationLevel,
        deepUncertaintyMethod,
        deepImageSize,
        deepNeighborhoodK,
        deepSimilaritySigma,
        synthetic: syntheticPayload()
      })
    });
    return pollJob<RunPayload>(job);
  }

  async function pollJob<T>(initialJob: JobStatusPayload): Promise<T> {
    let job = initialJob;
    setActiveJob(job);
    setLastJob(job);
    while (job.state === "queued" || job.state === "running" || job.state === "cancelling") {
      await delay(350);
      job = await apiFetch<JobStatusPayload>(`/api/jobs/${job.jobId}`);
      setActiveJob(job);
      setLastJob(job);
      setStatusText(job.stage);
    }
    setLastJob(job);
    if (job.state === "cancelled") {
      throw new Error("Job cancelled");
    }
    if (job.state === "error") {
      throw new Error(job.error ?? "Job failed");
    }
    if (!job.result) {
      throw new Error("Job completed without a result");
    }
    return job.result as T;
  }

  async function cancelActiveJob() {
    if (!activeJob || activeJob.state === "cancelling") return;
    try {
      const job = await apiFetch<JobStatusPayload>(`/api/jobs/${activeJob.jobId}/cancel`, {
        method: "POST"
      });
      setActiveJob(job);
      setLastJob(job);
      setStatusText("Cancelling job");
    } catch (error) {
      setStatusText(error instanceof Error ? error.message : "Cancel failed");
    }
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
          <button
            className={activeView === "experiments" ? "nav-item active" : "nav-item"}
            title="Experiments"
            onClick={() => setActiveView("experiments")}
          >
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
          <button
            className={activeView === "deep" ? "nav-item active" : "nav-item"}
            title="Deep Entropy"
            onClick={() => setActiveView("deep")}
          >
            <Layers size={18} />
            <span>Deep Entropy</span>
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

          <div className="setup-stage">
            <h3>Data</h3>
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
              <label>
                Preset
                <select value={syntheticPreset} onChange={(event) => applySyntheticPreset(event.target.value)}>
                  {syntheticPresets.map((preset) => (
                    <option value={preset.id} key={preset.id}>{preset.label}</option>
                  ))}
                </select>
              </label>
            )}
          </div>

          {dataset === "synthetic_shapes" && (
            <div className="setup-stage synthetic-controls">
              <h3>Synthetic controls</h3>
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

          <div className="setup-stage">
            <h3>Preprocessing</h3>
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
            <div className="readonly-setting">
              <span>Normalize</span>
              <strong>zero_one</strong>
            </div>
          </div>

          <div className="setup-stage">
            <h3>Representation</h3>
            <label>
              Color space
              <select value={representation} onChange={(event) => setRepresentation(event.target.value)}>
                <option value="rgb">RGB</option>
                <option value="grayscale">Gray</option>
                <option value="lab">Lab</option>
                <option value="red">Red</option>
                <option value="green">Green</option>
                <option value="blue">Blue</option>
              </select>
            </label>
          </div>

          <div className="setup-stage">
            <h3>Entropy</h3>
            <label>
              Measure
              <select value={entropyMeasure} onChange={(event) => setEntropyMeasure(event.target.value)}>
                <option value="shannon">Shannon</option>
                <option value="renyi" disabled>Renyi pending</option>
                <option value="tsallis" disabled>Tsallis pending</option>
              </select>
            </label>
            <label>
              Scope
              <select value={entropyScope} onChange={(event) => setEntropyScope(event.target.value)}>
                <option value="local">local</option>
                <option value="global" disabled>global pending</option>
                <option value="region" disabled>region pending</option>
              </select>
            </label>
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
          </div>

          <div className="setup-stage">
            <h3>Segmentation</h3>
            <label>
              Method
              <select value={segmentationMethod} onChange={(event) => setSegmentationMethod(event.target.value)}>
                {segmentationOptions.map((option) => (
                  <option value={option.value} key={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <div className="readonly-setting">
              <span>Features</span>
              <strong>{featureDescription(segmentationMethod)}</strong>
            </div>
          </div>

          <div className="setup-stage">
            <h3>Deep Entropy</h3>
            <div className="readonly-setting">
              <span>Backbone</span>
              <strong>{deepModelLabel(deepModel)}</strong>
            </div>
            <div className="readonly-setting">
              <span>Layer</span>
              <strong>{deepLayer}</strong>
            </div>
            <div className="readonly-setting">
              <span>Outputs</span>
              <strong>{deepMethodLabel(deepUncertaintyMethod)}</strong>
            </div>
          </div>

          <div className="setup-stage run-stage">
            <h3>Run</h3>
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

            <RunProgressPanel
              progress={activeProgress}
              compact
              canCancel={Boolean(activeJob && ["queued", "running"].includes(activeJob.state))}
              isCancelling={activeJob?.state === "cancelling"}
              onCancel={cancelActiveJob}
              onOpenDetails={() => setIsJobDetailsOpen(true)}
            />
          </div>

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
          <button className="secondary-action" onClick={() => setIsJobDetailsOpen(true)} disabled={!lastJob}>
            <Table2 size={17} />
            <span>Job Details</span>
          </button>
          <button className="secondary-action">
            <UploadCloud size={17} />
            <span>Attach Dataset</span>
          </button>
        </section>
      </aside>

      <section className="workspace">
        {activeView === "deep" ? (
          <DeepEntropyModule
            apiState={apiState}
            runResult={runResult}
            resultArtifacts={resultArtifacts}
            isRunning={isRunning}
            activeProgress={activeProgress}
            activeJob={activeJob}
            deepModel={deepModel}
            deepLayer={deepLayer}
            deepRepresentationLevel={deepRepresentationLevel}
            deepUncertaintyMethod={deepUncertaintyMethod}
            deepImageSize={deepImageSize}
            deepNeighborhoodK={deepNeighborhoodK}
            deepSimilaritySigma={deepSimilaritySigma}
            onDeepModelChange={(model) => {
              setDeepModel(model);
              const firstLayer = deepModelOptions.find((option) => option.value === model)?.layers[0];
              if (firstLayer) setDeepLayer(firstLayer);
            }}
            onDeepLayerChange={setDeepLayer}
            onDeepRepresentationLevelChange={setDeepRepresentationLevel}
            onDeepUncertaintyMethodChange={setDeepUncertaintyMethod}
            onDeepImageSizeChange={setDeepImageSize}
            onDeepNeighborhoodKChange={setDeepNeighborhoodK}
            onDeepSimilaritySigmaChange={setDeepSimilaritySigma}
            onRunDeep={runDeepPipeline}
            onCancelJob={cancelActiveJob}
            onOpenJobDetails={() => setIsJobDetailsOpen(true)}
          />
        ) : (
          <>
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

        <section className="evaluation-grid">
          <article className="surface secondary-evaluation">
            <div className="surface-heading">
              <BarChart3 size={18} />
              <h3>Secondary Evaluation</h3>
            </div>
            <dl className="secondary-metric-grid">
              {secondaryMetrics.map((metric) => (
                <div key={metric.label}>
                  <dt>{metric.label}</dt>
                  <dd>{metric.value}</dd>
                </div>
              ))}
            </dl>
          </article>

          <article className="surface confusion-panel">
            <div className="surface-heading">
              <Table2 size={18} />
              <h3>Confusion Matrix</h3>
            </div>
            <table className="confusion-matrix">
              <thead>
                <tr>
                  <th />
                  <th colSpan={2}>Predicted</th>
                </tr>
                <tr>
                  <th>Actual</th>
                  <th>Background</th>
                  <th>Foreground</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>BG</th>
                  <td>{formatCount(confusion.trueNegative)}</td>
                  <td>{formatCount(confusion.falsePositive)}</td>
                </tr>
                <tr>
                  <th>FG</th>
                  <td>{formatCount(confusion.falseNegative)}</td>
                  <td>{formatCount(confusion.truePositive)}</td>
                </tr>
              </tbody>
            </table>
          </article>
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
                <dd>{segmentationLabel(segmentationMethod)}</dd>
              </div>
              <div>
                <dt>Preset</dt>
                <dd>{dataset === "synthetic_shapes" ? syntheticPreset.replace(/_/g, " ") : "none"}</dd>
              </div>
              <div>
                <dt>Feature stack</dt>
                <dd>{featureDescription(segmentationMethod)}</dd>
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
                    <dt>Boundary F1</dt>
                    <dd>{formatMetric(run.metrics?.boundary_f1)}</dd>
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
              <div>
                <dt>Regions</dt>
                <dd>{runResult?.regions?.count ?? "none"}</dd>
              </div>
              <div>
                <dt>Graph edges</dt>
                <dd>{runResult?.regions?.edge_count ?? "none"}</dd>
              </div>
              <div>
                <dt>Spectral entropy</dt>
                <dd>{formatMetric(runResult?.graph?.normalized_spectral_entropy)}</dd>
              </div>
              <div>
                <dt>Partitions</dt>
                <dd>{runResult?.graph?.partition_count ?? "none"}</dd>
              </div>
              <div>
                <dt>Deep model</dt>
                <dd>{runResult?.deep?.model ?? "none"}</dd>
              </div>
              <div>
                <dt>Activation entropy</dt>
                <dd>{formatMetric(runResult?.deep?.mean_activation_entropy)}</dd>
              </div>
              <div>
                <dt>Latent entropy</dt>
                <dd>{formatMetric(runResult?.deep?.latent_entropy)}</dd>
              </div>
              <div>
                <dt>Predictive entropy</dt>
                <dd>{formatMetric(runResult?.deep?.predictive_entropy)}</dd>
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
  foreground: ${foregroundDescription(segmentationMethod)}
features:
  ${featureDescription(segmentationMethod)}
deep:
  backbone: ${deepModel}
  layer: ${deepLayer}
  representation: ${deepRepresentationLevel}
  method: ${deepUncertaintyMethod}
  neighborhood_k: ${deepNeighborhoodK}
  similarity_sigma: ${deepSimilaritySigma}
run_id:
  ${runIdPreview}_<timestamp>`}</pre>
          </article>
        </section>
          </>
        )}
      </section>

      <JobDetailsDrawer
        job={lastJob}
        fallbackRun={runResult}
        fallbackComparison={comparisonResult}
        open={isJobDetailsOpen}
        onClose={() => setIsJobDetailsOpen(false)}
        canCancel={Boolean(activeJob && ["queued", "running"].includes(activeJob.state))}
        isCancelling={activeJob?.state === "cancelling"}
        onCancel={cancelActiveJob}
      />
    </main>
  );
}

function RunProgressPanel({
  progress,
  compact = false,
  canCancel = false,
  isCancelling = false,
  onCancel,
  onOpenDetails
}: {
  progress: RunProgressModel | null;
  compact?: boolean;
  canCancel?: boolean;
  isCancelling?: boolean;
  onCancel?: () => void;
  onOpenDetails?: () => void;
}) {
  if (!progress) return null;

  return (
    <div className={compact ? "run-progress compact" : "run-progress"}>
      <div className="run-progress-header">
        <div>
          <strong>{progress.label}</strong>
          <span>{progress.currentStage}</span>
        </div>
        <time>{formatElapsed(progress.elapsedSeconds)}</time>
      </div>
      <div className="run-progress-track" aria-label={`${progress.label} progress`}>
        <span style={{ width: `${progress.percent}%` }} />
      </div>
      <div className="run-progress-meta">
        <span>{Math.round(progress.percent)}%</span>
        <span>{compact ? "Waiting for API" : "API will mark complete when artifacts are saved"}</span>
      </div>
      <div className="run-progress-actions">
        <button className="secondary-action compact" onClick={onOpenDetails} type="button">
          <Table2 size={15} />
          <span>Details</span>
        </button>
        <button
          className="secondary-action compact danger"
          onClick={onCancel}
          disabled={!canCancel || isCancelling}
          type="button"
        >
          <XCircle size={15} />
          <span>{isCancelling ? "Cancelling" : "Cancel"}</span>
        </button>
      </div>
      {!compact && (
        <div className="run-progress-stages">
          {progress.stages.map((stage, index) => (
            <span
              className={
                index < progress.activeStageIndex
                  ? "complete"
                  : index === progress.activeStageIndex
                    ? "active"
                    : ""
              }
              key={stage}
            >
              {stage}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function JobDetailsDrawer({
  job,
  fallbackRun,
  fallbackComparison,
  open,
  onClose,
  canCancel = false,
  isCancelling = false,
  onCancel
}: {
  job: JobStatusPayload | null;
  fallbackRun: RunPayload | null;
  fallbackComparison: ComparisonPayload | null;
  open: boolean;
  onClose: () => void;
  canCancel?: boolean;
  isCancelling?: boolean;
  onCancel?: () => void;
}) {
  if (!open) return null;
  const result = job?.result ?? (job?.kind === "comparison" ? fallbackComparison : fallbackRun);
  const outputDirectory = result && "outputDirectory" in result ? result.outputDirectory : undefined;
  const runtimeSeconds = result?.runtime?.duration_seconds;
  const timeline = job?.timeline ?? [];

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="job-drawer" aria-label="Run details">
        <div className="drawer-heading">
          <div>
            <p className="eyebrow">Job Details</p>
            <h2>{job ? jobLabel(job) : "No job selected"}</h2>
          </div>
          <button className="icon-action" onClick={onClose} type="button" title="Close details">
            <XCircle size={18} />
          </button>
        </div>

        {job ? (
          <>
            <div className="drawer-actions">
              <button
                className="secondary-action danger"
                onClick={onCancel}
                disabled={!canCancel || isCancelling}
                type="button"
              >
                <XCircle size={16} />
                <span>{isCancelling ? "Cancelling" : "Cancel Job"}</span>
              </button>
            </div>

            <dl className="drawer-grid">
              <div>
                <dt>State</dt>
                <dd>{job.state}</dd>
              </div>
              <div>
                <dt>Stage</dt>
                <dd>{job.stage}</dd>
              </div>
              <div>
                <dt>Progress</dt>
                <dd>{Math.round(job.percent)}%</dd>
              </div>
              <div>
                <dt>Elapsed</dt>
                <dd>{formatElapsed(job.elapsedSeconds)}</dd>
              </div>
              <div>
                <dt>Started</dt>
                <dd>{formatTimestamp(job.startedAt)}</dd>
              </div>
              <div>
                <dt>Updated</dt>
                <dd>{formatTimestamp(job.updatedAt)}</dd>
              </div>
            </dl>

            {job.error && (
              <div className={job.state === "cancelled" ? "drawer-note" : "drawer-note error"}>
                {job.error}
              </div>
            )}

            <section className="drawer-section">
              <h3>Output</h3>
              <dl className="drawer-list">
                <div>
                  <dt>Experiment</dt>
                  <dd>{result?.experiment ?? "pending"}</dd>
                </div>
                <div>
                  <dt>Folder</dt>
                  <dd>{outputDirectory ?? "pending"}</dd>
                </div>
                <div>
                  <dt>Runtime</dt>
                  <dd>{formatRuntime(runtimeSeconds)}</dd>
                </div>
              </dl>
            </section>

            <section className="drawer-section">
              <h3>Timeline</h3>
              <div className="job-timeline">
                {timeline.length > 0 ? (
                  timeline.map((entry) => (
                    <div key={`${entry.stage}-${entry.startedAt}`}>
                      <span />
                      <strong>{entry.stage}</strong>
                      <em>{Math.round(entry.percent)}%</em>
                      <time>{formatRuntime(entry.durationSeconds)}</time>
                    </div>
                  ))
                ) : (
                  <p>No stages recorded yet</p>
                )}
              </div>
            </section>

            <section className="drawer-section">
              <h3>Run Metadata</h3>
              <pre>{formatJobMetadata(result)}</pre>
            </section>
          </>
        ) : (
          <div className="drawer-note">Start a run to see job status and timing here.</div>
        )}
      </aside>
    </>
  );
}

function DeepEntropyModule({
  apiState,
  runResult,
  resultArtifacts,
  isRunning,
  activeProgress,
  activeJob,
  deepModel,
  deepLayer,
  deepRepresentationLevel,
  deepUncertaintyMethod,
  deepImageSize,
  deepNeighborhoodK,
  deepSimilaritySigma,
  onDeepModelChange,
  onDeepLayerChange,
  onDeepRepresentationLevelChange,
  onDeepUncertaintyMethodChange,
  onDeepImageSizeChange,
  onDeepNeighborhoodKChange,
  onDeepSimilaritySigmaChange,
  onRunDeep,
  onCancelJob,
  onOpenJobDetails
}: {
  apiState: ApiState;
  runResult: RunPayload | null;
  resultArtifacts: Record<string, string>;
  isRunning: boolean;
  activeProgress: RunProgressModel | null;
  activeJob: JobStatusPayload | null;
  deepModel: string;
  deepLayer: string;
  deepRepresentationLevel: string;
  deepUncertaintyMethod: string;
  deepImageSize: number;
  deepNeighborhoodK: number;
  deepSimilaritySigma: number;
  onDeepModelChange: (value: string) => void;
  onDeepLayerChange: (value: string) => void;
  onDeepRepresentationLevelChange: (value: string) => void;
  onDeepUncertaintyMethodChange: (value: string) => void;
  onDeepImageSizeChange: (value: number) => void;
  onDeepNeighborhoodKChange: (value: number) => void;
  onDeepSimilaritySigmaChange: (value: number) => void;
  onRunDeep: () => void;
  onCancelJob: () => void;
  onOpenJobDetails: () => void;
}) {
  const deep = runResult?.deep;
  const model = deepModelOptions.find((option) => option.value === deepModel) ?? deepModelOptions[1];
  const representationOption =
    featureRepresentationOptions.find((option) => option.value === deepRepresentationLevel) ??
    featureRepresentationOptions[0];
  const methodOption =
    uncertaintyMethodOptions.find((option) => option.value === deepUncertaintyMethod) ??
    uncertaintyMethodOptions[0];
  const deepMetrics = [
    { label: "Activation entropy", value: deep?.mean_activation_entropy },
    { label: "Fuzzy entropy", value: deep?.mean_fuzzy_entropy },
    { label: "Rough uncertainty", value: deep?.mean_rough_uncertainty },
    { label: "Fuzzy-rough", value: deep?.mean_fuzzy_rough_uncertainty },
    { label: "Latent entropy", value: deep?.latent_entropy },
    { label: "Predictive entropy", value: deep?.predictive_entropy },
    { label: "Classes", value: deep?.class_count, count: true }
  ];

  return (
    <>
      <header className="topbar">
        <div>
          <p className="eyebrow">Deep Representations</p>
          <h2>Deep Entropy</h2>
        </div>
        <span className={deep?.available ? "status-pill ready" : "status-pill missing"}>
          {deep?.available ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
          {deep?.available ? "Deep outputs ready" : "Deep outputs pending"}
        </span>
      </header>

      <section className="deep-control-grid">
        <article className="surface deep-run-panel">
          <div className="surface-heading split">
            <div className="surface-title">
              <Layers size={18} />
              <h3>Feature Extraction</h3>
            </div>
            <button
              className="primary-action compact-primary"
              onClick={onRunDeep}
              disabled={apiState !== "online" || isRunning}
            >
              <Play size={17} />
              <span>{isRunning ? "Running" : "Run Deep Slice"}</span>
            </button>
          </div>

          <div className="deep-control-fields">
            <label>
              Model
              <select value={deepModel} onChange={(event) => onDeepModelChange(event.target.value)}>
                {deepModelOptions.map((option) => (
                  <option value={option.value} key={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              Selected layer
              <select value={deepLayer} onChange={(event) => onDeepLayerChange(event.target.value)}>
                {model.layers.map((layer) => (
                  <option value={layer} key={layer}>{layer}</option>
                ))}
              </select>
            </label>
            <label>
              Feature representation
              <select
                value={deepRepresentationLevel}
                onChange={(event) => onDeepRepresentationLevelChange(event.target.value)}
              >
                {featureRepresentationOptions.map((option) => (
                  <option value={option.value} key={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              Uncertainty method
              <select
                value={deepUncertaintyMethod}
                onChange={(event) => onDeepUncertaintyMethodChange(event.target.value)}
              >
                {uncertaintyMethodOptions.map((option) => (
                  <option value={option.value} key={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              Image size
              <input
                type="number"
                value={deepImageSize}
                min={48}
                max={512}
                step={16}
                onChange={(event) => onDeepImageSizeChange(Number(event.target.value))}
              />
            </label>
            <label>
              k neighbors
              <input
                type="number"
                value={deepNeighborhoodK}
                min={1}
                max={64}
                onChange={(event) => onDeepNeighborhoodKChange(Number(event.target.value))}
              />
            </label>
            <label>
              RBF sigma
              <input
                type="number"
                value={deepSimilaritySigma}
                min={0}
                max={5}
                step={0.05}
                onChange={(event) => onDeepSimilaritySigmaChange(Number(event.target.value))}
              />
            </label>
          </div>

          <RunProgressPanel
            progress={activeProgress}
            canCancel={Boolean(activeJob && ["queued", "running"].includes(activeJob.state))}
            isCancelling={activeJob?.state === "cancelling"}
            onCancel={onCancelJob}
            onOpenDetails={onOpenJobDetails}
          />
        </article>

        <article className="surface deep-summary-panel">
          <div className="surface-heading">
            <Activity size={18} />
            <h3>Current Tensor</h3>
          </div>
          <dl className="parameter-list">
            <div>
              <dt>Model</dt>
              <dd>{deep?.model ?? deepModel}</dd>
            </div>
            <div>
              <dt>Layer</dt>
              <dd>{deep?.layer ?? deepLayer}</dd>
            </div>
            <div>
              <dt>Feature tensor</dt>
              <dd>{formatDimensions(deep?.feature_shape)}</dd>
            </div>
            <div>
              <dt>Embedding</dt>
              <dd>{deep?.latent_size ?? "pending"}</dd>
            </div>
            <div>
              <dt>Basis</dt>
              <dd>{representationOption.label}</dd>
            </div>
            <div>
              <dt>Method</dt>
              <dd>{deepMethodLabel(deep?.uncertainty_method ?? deepUncertaintyMethod)}</dd>
            </div>
            <div>
              <dt>Neighborhood</dt>
              <dd>k {deep?.neighborhood_k ?? deepNeighborhoodK}</dd>
            </div>
            <div>
              <dt>RBF sigma</dt>
              <dd>{deep?.similarity_sigma == null && deepSimilaritySigma === 0 ? "auto" : formatMetric(deep?.similarity_sigma ?? deepSimilaritySigma)}</dd>
            </div>
          </dl>
        </article>
      </section>

      <section className="status-strip deep-status-strip">
        {deepPipelineStages.map((stage, index) => (
          <div className="stage" key={stage}>
            <span className={deep?.available || index < 4 ? "stage-dot complete" : "stage-dot"} />
            <span>{stage}</span>
          </div>
        ))}
      </section>

      <section className="deep-artifact-grid">
        <article className="surface large">
          <div className="surface-heading">
            <Image size={18} />
            <h3>Deep Feature Maps</h3>
          </div>
          <div className="artifact-board">
            <ImagePanel title="Feature map" src={resultArtifacts.deep_feature_map} />
            <ImagePanel title="Activation entropy" src={resultArtifacts.activation_entropy} />
          </div>
        </article>

        <article className="surface">
          <div className="surface-heading">
            <BarChart3 size={18} />
            <h3>Entropy Scores</h3>
          </div>
          <dl className="secondary-metric-grid deep-metric-grid">
            {deepMetrics.map((metric) => (
              <div key={metric.label}>
                <dt>{metric.label}</dt>
                <dd>{metric.count ? formatCount(metric.value) : formatMetric(metric.value)}</dd>
              </div>
            ))}
          </dl>
        </article>

        <article className="surface">
          <div className="surface-heading">
            <Boxes size={18} />
            <h3>Embeddings</h3>
          </div>
          <div className="mini-artifacts">
            <ImagePanel title="Latent entropy" src={resultArtifacts.latent_entropy} />
            <ImagePanel title="Predictive entropy" src={resultArtifacts.predictive_entropy} />
          </div>
        </article>
      </section>

      <section className="deep-artifact-grid">
        <article className="surface">
          <div className="surface-heading">
            <Activity size={18} />
            <h3>Fuzzy Analysis</h3>
          </div>
          <div className="mini-artifacts">
            <ImagePanel title="Fuzzy entropy" src={resultArtifacts.fuzzy_entropy} />
            <ImagePanel title="Activation entropy" src={resultArtifacts.activation_entropy} />
          </div>
        </article>

        <article className="surface">
          <div className="surface-heading">
            <Network size={18} />
            <h3>Rough Analysis</h3>
          </div>
          <div className="mini-artifacts">
            <ImagePanel title="Rough uncertainty" src={resultArtifacts.rough_uncertainty} />
            <ImagePanel title="Fuzzy-rough uncertainty" src={resultArtifacts.fuzzy_rough_uncertainty} />
          </div>
        </article>
      </section>

      <section className="deep-method-grid">
        {uncertaintyMethodOptions.map((option) => (
          <article
            className={option.value === deepUncertaintyMethod ? "method-card active" : "method-card"}
            key={option.value}
          >
            <strong>{option.label}</strong>
            <span>{option.output}</span>
          </article>
        ))}
      </section>

      <section className="content-grid">
        <article className="surface">
          <div className="surface-heading">
            <Network size={18} />
            <h3>Representation Pipeline</h3>
          </div>
          <div className="deep-flow">
            <div>Image</div>
            <div>{model.label}</div>
            <div>{deepLayer}</div>
            <div>{representationOption.label}</div>
            <div>{methodOption.label}</div>
            <div>Evaluation</div>
          </div>
        </article>

        <article className="surface">
          <div className="surface-heading">
            <Table2 size={18} />
            <h3>Predictive Distribution</h3>
          </div>
          <div className="probability-list">
            {(deep?.top_probabilities ?? []).map((item) => (
              <div key={item.index}>
                <span>class {item.index}</span>
                <meter min={0} max={1} value={item.probability} />
                <strong>{formatProbability(item.probability)}</strong>
              </div>
            ))}
            {(deep?.top_probabilities ?? []).length === 0 && (
              <div className="empty-comparison">
                <BarChart3 size={22} />
                <span>No predictive distribution yet</span>
              </div>
            )}
          </div>
        </article>
      </section>
    </>
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

function formatCount(value?: number) {
  return value == null ? "0" : Math.round(value).toLocaleString();
}

function formatDimensions(value?: number[]) {
  return value && value.length > 0 ? value.join(" x ") : "pending";
}

function formatProbability(value?: number) {
  return value == null ? "0.0%" : `${(value * 100).toFixed(1)}%`;
}

function formatRuntime(value?: number | null) {
  return value == null ? "unknown" : `${value.toFixed(2)} s`;
}

function formatTimestamp(value?: number | null) {
  if (value == null) return "unknown";
  return new Date(value * 1000).toLocaleTimeString();
}

function jobLabel(job: JobStatusPayload) {
  const shortId = job.jobId.slice(0, 8);
  if (job.kind === "deep") return `Deep job ${shortId}`;
  if (job.kind === "comparison") return `Comparison job ${shortId}`;
  return `Slice job ${shortId}`;
}

function formatJobMetadata(result?: RunPayload | ComparisonPayload | null) {
  if (!result) return "pending";
  if ("runMetadata" in result && result.runMetadata) {
    return JSON.stringify(result.runMetadata, null, 2);
  }
  if ("parameters" in result) {
    return JSON.stringify(
      {
        sampleId: result.sampleId,
        parameters: result.parameters,
        bestVariantId: result.bestVariantId,
      },
      null,
      2,
    );
  }
  return JSON.stringify(result, null, 2);
}

function formatEntropy(run: RunPayload) {
  const entropy = run.runMetadata?.entropy;
  if (!entropy) return "unknown";
  return `${entropy.scope ?? "local"} ${entropy.name ?? "shannon"}`;
}

function formatSegmentation(run: RunPayload) {
  const name = run.runMetadata?.segmentation?.name;
  if (!name) return "unknown";
  return segmentationLabel(name);
}

function segmentationLabel(name: string) {
  const match = segmentationOptions.find((option) => option.value === name);
  if (match) return match.label;
  if (name === "maximum_entropy_threshold") return "Kapur maximum entropy";
  if (name === "gmm") return "Gaussian mixture";
  if (name === "rf") return "Random Forest";
  return name.replace(/_/g, " ");
}

function featureDescription(method: string) {
  if (featureStackMethods.has(method)) return "[grayscale, local_entropy, gradient_magnitude]";
  if (regionMethods.has(method)) return "grayscale + gradient/region labels";
  if (method === "kapur" || method === "maximum_entropy_threshold") return "entropy_map";
  return "grayscale intensity";
}

function foregroundDescription(method: string) {
  if (maskOverlapMethods.has(method)) return "mask_overlap_eval";
  if (method === "random_forest") return "mask_supervised_train";
  return "high";
}

function buildRunProgress(activeRun: ActiveRunState, elapsedSeconds: number): RunProgressModel {
  const profile = runProgressProfiles[activeRun.kind];
  const smoothProgress = 100 * (1 - Math.exp(-elapsedSeconds / Math.max(profile.expectedSeconds, 1)));
  const percent = Math.max(3, Math.min(94, smoothProgress));
  const activeStageIndex = Math.min(
    profile.stages.length - 1,
    Math.floor((percent / 100) * profile.stages.length)
  );
  return {
    label: profile.label,
    percent,
    elapsedSeconds,
    currentStage: profile.stages[activeStageIndex],
    stages: profile.stages,
    activeStageIndex
  };
}

function buildJobProgress(job: JobStatusPayload): RunProgressModel {
  const profile = runProgressProfiles[job.kind] ?? runProgressProfiles.slice;
  const safeTotal = Math.max(1, job.totalStages);
  const activeStageIndex = Math.max(0, Math.min(safeTotal - 1, job.stageIndex));
  return {
    label: profile.label,
    percent: Math.max(0, Math.min(100, job.percent)),
    elapsedSeconds: job.elapsedSeconds,
    currentStage: job.stage,
    stages: buildStageLabels(profile.stages, job.stage, safeTotal, activeStageIndex),
    activeStageIndex
  };
}

function buildStageLabels(profileStages: string[], stage: string, totalStages: number, activeStageIndex: number) {
  if (profileStages.length === totalStages) {
    return profileStages.map((label, index) => (index === activeStageIndex ? stage : label));
  }
  return Array.from({ length: totalStages }, (_, index) => {
    if (index === activeStageIndex) return stage;
    return profileStages[index] ?? `Stage ${index + 1}`;
  });
}

function formatElapsed(seconds: number) {
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s`;
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remaining}`;
}

function delay(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function deepModelLabel(model: string) {
  return deepModelOptions.find((option) => option.value === model)?.label ?? model;
}

function deepMethodLabel(method: string) {
  return uncertaintyMethodOptions.find((option) => option.value === method)?.label ?? method;
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
