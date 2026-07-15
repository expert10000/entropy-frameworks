import {
  Activity,
  BarChart3,
  Boxes,
  Braces,
  Database,
  Download,
  FlaskConical,
  Gauge,
  GitBranch,
  Image,
  LineChart,
  Network,
  Play,
  RefreshCw,
  Save,
  Settings2,
  SlidersHorizontal
} from "lucide-react";

type Metric = {
  label: string;
  value: string;
  delta: string;
};

const metrics: Metric[] = [
  { label: "Mean IoU", value: "0.81", delta: "+0.06" },
  { label: "Dice", value: "0.89", delta: "+0.04" },
  { label: "Pixel accuracy", value: "0.93", delta: "+0.02" },
  { label: "Entropy-error r", value: "0.42", delta: "+0.11" }
];

const pipelineStages = [
  "Dataset",
  "Preprocessing",
  "Representation",
  "Entropy",
  "Segmentation",
  "Evaluation",
  "Report"
];

const experiments = [
  "e01_synthetic_shannon",
  "pet_superpixel_shannon_graph",
  "small_data_rgb_entropy"
];

function App() {
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
            Experiment
            <select defaultValue="e01_synthetic_shannon">
              {experiments.map((experiment) => (
                <option key={experiment}>{experiment}</option>
              ))}
            </select>
          </label>

          <label>
            Dataset
            <select defaultValue="synthetic_shapes">
              <option>synthetic_shapes</option>
              <option>skimage_examples</option>
              <option>oxford_iiit_pet</option>
            </select>
          </label>

          <label>
            Representation
            <select defaultValue="grayscale">
              <option>rgb</option>
              <option>grayscale</option>
              <option>lab</option>
              <option>slic_superpixels</option>
            </select>
          </label>

          <label>
            Entropy Measure
            <select defaultValue="shannon">
              <option>shannon</option>
              <option>renyi</option>
              <option>tsallis</option>
            </select>
          </label>

          <div className="two-column">
            <label>
              Bins
              <input type="number" defaultValue={64} min={2} max={512} />
            </label>
            <label>
              Window
              <input type="number" defaultValue={9} min={3} max={31} step={2} />
            </label>
          </div>

          <button className="primary-action">
            <Play size={18} />
            <span>Run Pipeline</span>
          </button>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">VE-0.1 Classical Vertical Slice</p>
            <h2>Entropy-Guided Image Segmentation</h2>
          </div>
          <div className="topbar-actions">
            <button title="Refresh">
              <RefreshCw size={18} />
            </button>
            <button title="Save configuration">
              <Save size={18} />
            </button>
            <button title="Export results">
              <Download size={18} />
            </button>
          </div>
        </header>

        <section className="status-strip">
          {pipelineStages.map((stage, index) => (
            <div className="stage" key={stage}>
              <span className={index < 4 ? "stage-dot complete" : "stage-dot"} />
              <span>{stage}</span>
            </div>
          ))}
        </section>

        <section className="metrics-grid">
          {metrics.map((metric) => (
            <article className="metric-card" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
              <em>{metric.delta}</em>
            </article>
          ))}
        </section>

        <section className="content-grid">
          <article className="surface large">
            <div className="surface-heading">
              <Image size={18} />
              <h3>Image And Entropy Map</h3>
            </div>
            <div className="image-comparison">
              <div className="synthetic-preview original" aria-label="Original synthetic sample" />
              <div className="synthetic-preview entropy" aria-label="Entropy heatmap preview" />
            </div>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <GitBranch size={18} />
              <h3>Segmentation</h3>
            </div>
            <div className="segmentation-preview">
              <span />
              <span />
              <span />
              <span />
            </div>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <LineChart size={18} />
              <h3>Run Trend</h3>
            </div>
            <div className="trend">
              <span style={{ height: "34%" }} />
              <span style={{ height: "52%" }} />
              <span style={{ height: "41%" }} />
              <span style={{ height: "68%" }} />
              <span style={{ height: "76%" }} />
              <span style={{ height: "61%" }} />
              <span style={{ height: "84%" }} />
            </div>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <SlidersHorizontal size={18} />
              <h3>Parameters</h3>
            </div>
            <dl className="parameter-list">
              <div>
                <dt>Seed</dt>
                <dd>42</dd>
              </div>
              <div>
                <dt>Image size</dt>
                <dd>256 x 256</dd>
              </div>
              <div>
                <dt>Superpixels</dt>
                <dd>250</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>outputs/runs</dd>
              </div>
            </dl>
          </article>

          <article className="surface">
            <div className="surface-heading">
              <Braces size={18} />
              <h3>Config Preview</h3>
            </div>
            <pre>{`entropy:
  name: shannon
  parameters:
    bins: 64
segmentation:
  name: maximum_entropy_threshold`}</pre>
          </article>
        </section>
      </section>
    </main>
  );
}

export default App;
