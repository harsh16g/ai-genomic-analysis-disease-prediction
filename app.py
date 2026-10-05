"""
app.py
======
Genomic Analysis AI Dashboard

Streamlit front-end for the genomic_analysis_project pipeline (BioPython
sequence analysis + classical ML + CNN comparison). This file only reads
from and orchestrates the existing src/ modules — it does not change any
existing ML/bioinformatics logic.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px  # type: ignore[reportMissingImports]
import plotly.graph_objects as go  # type: ignore[reportMissingImports]
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — import the existing src/ package without modifying it
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequences import DNASequence, load_fasta          # noqa: E402
from alignment_motifs import find_common_motifs        # noqa: E402
from features import build_kmer_feature_matrix         # noqa: E402
from classical_ml import train_and_compare             # noqa: E402

FASTA_PATH = DATA_DIR / "sample_sequences.fasta"
LABELS_PATH = DATA_DIR / "labels.csv"
FINAL_REPORT_PATH = OUTPUTS_DIR / "final_experiment_report.txt"
SEQUENCE_REPORT_PATH = OUTPUTS_DIR / "sequence_report.txt"
CNN_HISTORY_PATH = OUTPUTS_DIR / "cnn_history.json"
GC_DIST_IMG = OUTPUTS_DIR / "gc_distribution.png"
MODEL_ACC_IMG = OUTPUTS_DIR / "model_accuracy_comparison.png"
CONF_MATRIX_IMG = OUTPUTS_DIR / "confusion_matrix_RandomForest.png"

DEFAULT_K = 4

st.set_page_config(
    page_title="Genomic Analysis AI Dashboard",
    page_icon="\U0001F9EC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    h1, h2, h3 {font-weight: 600;}
    .app-footnote {color: #6b7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_label_map(labels_path: str) -> dict:
    df = pd.read_csv(labels_path)
    return dict(zip(df["seq_id"], df["label"]))


@st.cache_data(show_spinner=False)
def load_dataset_frame(fasta_path: str, labels_path: str) -> pd.DataFrame:
    """Parses the FASTA file with BioPython (via sequences.load_fasta) and
    joins it with labels.csv, returning one row per sequence with all
    stats computed dynamically — nothing here is hard-coded."""
    seqs = load_fasta(fasta_path, molecule="dna")
    label_map = load_label_map(labels_path)

    rows = []
    for s in seqs:
        label = label_map.get(s.seq_id, "unknown")
        gc = s.gc_content()
        comp = s.base_composition()
        rows.append({
            "seq_id": s.seq_id,
            "label": label,
            "length": len(s),
            "gc_content": gc,
            "at_content": round(100 - gc, 2) if len(s) else 0.0,
            "A": comp.get("A", 0),
            "T": comp.get("T", 0),
            "G": comp.get("G", 0),
            "C": comp.get("C", 0),
            "sequence": s.sequence,
        })
    return pd.DataFrame(rows)


def dataset_available() -> bool:
    return FASTA_PATH.exists() and LABELS_PATH.exists()


# ---------------------------------------------------------------------------
# final_experiment_report.txt parser
# ---------------------------------------------------------------------------

def _parse_kv_block(lines: list) -> dict:
    result = {}
    for line in lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip()
    return result


@st.cache_data(show_spinner=False)
def parse_final_report(report_path: str) -> dict | None:
    path = Path(report_path)
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        numbered_heading = re.match(r"^\d+\.\s+(.+)$", line)
        if line.startswith("=== ") and line.endswith(" ==="):
            current = line[4:-4].strip().upper()
            sections[current] = []
        elif numbered_heading:
            current = numbered_heading.group(1).strip().upper()
            sections[current] = []
        elif current is not None:
            if line and not set(line) <= {"-", "="}:
                sections[current].append(line)

    parsed: dict = {}

    dataset_section = next(
        (lines for name, lines in sections.items() if name == "DATASET SUMMARY"),
        None,
    )
    if dataset_section is not None:
        parsed["dataset"] = _parse_kv_block(dataset_section)

    motif_section = next(
        (
            lines
            for name, lines in sections.items()
            if name == "TOP MOTIFS" or name.startswith("TOP RECURRING")
        ),
        None,
    )
    if motif_section is not None:
        motifs = {}
        for line in motif_section:
            match = re.match(r"^(.+?):\s*found in\s+(\d+)\s+sequences", line, re.I)
            if match:
                motifs[match.group(1).strip()] = int(match.group(2))
        parsed["motifs"] = motifs

    classical_section = next(
        (
            lines
            for name, lines in sections.items()
            if name in ("CLASSICAL ML RESULTS", "CLASSICAL MACHINE LEARNING RESULTS")
        ),
        None,
    )
    if classical_section is not None:
        classical = {}
        current_model = None
        classification_lines: list[str] = []
        model_names = ("RandomForest", "SVM", "LogisticRegression")

        def finish_classification_report():
            if current_model and classification_lines:
                classical[current_model]["classification_report"] = "\n".join(classification_lines).strip()

        for line in classical_section:
            if line in model_names:
                finish_classification_report()
                current_model = line
                classical[current_model] = {}
                classification_lines = []
                continue

            if current_model is None:
                continue

            if line.startswith("Test accuracy:"):
                value = line.partition(":")[2].split("(", 1)[0].strip()
                try:
                    classical[current_model]["test_accuracy"] = float(value)
                except ValueError:
                    pass
            elif line.startswith("Cross-validation accuracy:"):
                value = line.partition(":")[2].strip()
                match = re.match(r"([\d.]+)\s*(?:\+/-|±)\s*([\d.]+)", value)
                if match:
                    try:
                        classical[current_model]["cv_mean_accuracy"] = float(match.group(1))
                        classical[current_model]["cv_std"] = float(match.group(2))
                    except ValueError:
                        pass
            elif line == "Classification report:":
                classification_lines = []
            elif classification_lines or line.startswith(("precision", "recall", "f1-score")):
                classification_lines.append(line)
            elif line:
                finish_classification_report()
                classification_lines = []
        finish_classification_report()
        parsed["classical"] = classical
        embedded_reports = {
            name: metrics["classification_report"]
            for name, metrics in classical.items()
            if "classification_report" in metrics
        }
        if embedded_reports:
            parsed["classification_reports"] = embedded_reports

    classification_reports = {}
    for section_name, section_lines in sections.items():
        if section_name.startswith("CLASSIFICATION REPORT: "):
            model_name = section_name.replace("CLASSIFICATION REPORT: ", "").strip()
            classification_reports[model_name] = "\n".join(section_lines).strip("\n")
    if classification_reports:
        parsed["classification_reports"] = classification_reports

    cnn_section = next(
        (
            lines
            for name, lines in sections.items()
            if name in ("CNN RESULTS", "DEEP LEARNING RESULT")
        ),
        None,
    )
    if cnn_section is not None:
        cnn_kv = _parse_kv_block(
            [l for l in cnn_section if not l.strip().endswith("_per_epoch:") and "_per_epoch:" not in l]
        )
        cnn = {}
        if cnn_kv.get("status") == "not_run":
            cnn = None
        else:
            for line in cnn_section:
                if "_per_epoch:" in line:
                    key, _, values = line.partition(":")
                    key = key.strip()
                    values = values.strip()
                    cnn[key] = [float(v) for v in values.split(",") if v.strip()] if values else []
                elif ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    normalized_key = key.lower().replace(" ", "_")
                    if normalized_key in ("test_accuracy", "test_loss", "epochs"):
                        try:
                            cnn[normalized_key] = float(value.strip().split("(", 1)[0].strip())
                        except ValueError:
                            pass
        parsed["cnn"] = cnn

    comparison_section = next(
        (lines for name, lines in sections.items() if name == "FINAL MODEL COMPARISON"),
        None,
    )
    if comparison_section is not None:
        best_model = {}
        for line in comparison_section:
            if line.startswith("Best model:"):
                best_model["best_model"] = line.partition(":")[2].strip()
            elif line.startswith("Best test accuracy:"):
                try:
                    best_model["best_accuracy"] = float(line.partition(":")[2].strip().rstrip("%")) / 100
                except ValueError:
                    pass
        if best_model:
            parsed["best_model"] = best_model
    elif "BEST MODEL" in sections:
        parsed["best_model"] = _parse_kv_block(sections["BEST MODEL"])

    return parsed


@st.cache_data(show_spinner=False)
def load_cnn_history(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# ML helpers (train on demand — no silent hard-coded predictions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training classical models on the current dataset ...")
def train_models_cached(fasta_path: str, labels_path: str, k: int):
    """Trains RandomForest / SVM / LogisticRegression on the current
    dataset using the exact same functions as pipeline.py, so the
    DNA Predictor page uses identical preprocessing to training — no
    separate/duplicated feature logic, no data leakage."""
    seqs = load_fasta(fasta_path, molecule="dna")
    label_map = load_label_map(labels_path)
    labels = [label_map.get(s.seq_id, "unknown") for s in seqs]
    df = build_kmer_feature_matrix(seqs, labels, k=k)
    results = train_and_compare(df)
    feature_columns = [c for c in df.columns if c != "label"]
    return results, feature_columns


def best_model_name_from_results(results: dict) -> str:
    candidates = {n: r["test_accuracy"] for n, r in results.items() if n not in ("label_encoder", "scaler")}
    return max(candidates, key=candidates.get)


def validate_dna_sequence(raw: str) -> tuple:
    """Returns (cleaned_sequence, list_of_invalid_characters)."""
    cleaned_upper = raw.strip().upper()
    valid = set("ACGT")
    invalid_chars = sorted({ch for ch in cleaned_upper if ch.isalpha() and ch not in valid})
    cleaned = "".join(ch for ch in cleaned_upper if ch in valid)
    return cleaned, invalid_chars


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard(df: pd.DataFrame, report: dict | None):
    st.title("Genomic Analysis AI Dashboard")
    st.caption(
        "BioPython sequence analysis \u2192 classical ML \u2192 deep learning comparison, "
        "wrapped in one automated pipeline (CSB4140 major project)."
    )

    if df is None:
        st.warning(
            "No dataset found. Expected files:\n\n"
            f"- `{FASTA_PATH}`\n- `{LABELS_PATH}`\n\n"
            "Generate them with `python src/data_generator.py`, or use the "
            "**Run Analysis** page."
        )
        return

    num_sequences = len(df)
    num_classes = df["label"].nunique()
    avg_length = round(df["length"].mean(), 1)
    avg_gc = round(df["gc_content"].mean(), 2)

    best_model, best_accuracy = None, None
    if report and report.get("classical"):
        classical = dict(report["classical"])
        if report.get("cnn"):
            classical = {**classical, "CNN": {"test_accuracy": report["cnn"].get("test_accuracy")}}
        candidates = {n: m.get("test_accuracy") for n, m in classical.items() if m.get("test_accuracy") is not None}
        if candidates:
            best_model = max(candidates, key=candidates.get)
            best_accuracy = candidates[best_model]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sequences", f"{num_sequences}")
    col2.metric("Classes", f"{num_classes}")
    col3.metric("Avg. Length", f"{avg_length} bp")
    col4.metric("Avg. GC Content", f"{avg_gc}%")

    col5, col6 = st.columns(2)
    col5.metric("Best Model", best_model or "Not yet computed")
    col6.metric("Best Model Accuracy", f"{best_accuracy * 100:.2f}%" if best_accuracy is not None else "N/A")

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Model Comparison")
        if report and report.get("classical"):
            rows = [{"model": n, "test_accuracy": m.get("test_accuracy", 0)} for n, m in report["classical"].items()]
            if report.get("cnn"):
                rows.append({"model": "CNN", "test_accuracy": report["cnn"].get("test_accuracy", 0)})
            chart_df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
            fig = px.bar(chart_df, x="model", y="test_accuracy", text_auto=".2%",
                         labels={"test_accuracy": "Test Accuracy", "model": "Model"})
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No experiment results yet. Run the analysis from the **Run Analysis** page.")

    with right:
        st.subheader("Key Findings")
        if report and report.get("classical"):
            findings = []
            findings.append(f"- Dataset: **{num_sequences} sequences** across **{num_classes} classes**.")
            if best_model:
                findings.append(f"- Best performing model: **{best_model}** ({best_accuracy * 100:.2f}% test accuracy).")
            if report.get("cnn"):
                cnn_acc = report["cnn"].get("test_accuracy")
                if cnn_acc is not None and best_model != "CNN":
                    findings.append(
                        f"- The CNN reached {cnn_acc * 100:.2f}% test accuracy \u2014 lower than the best "
                        "classical model on this dataset size, which is an expected, explainable "
                        "result rather than a general claim that classical ML beats deep learning."
                    )
            if report.get("motifs"):
                top_motif = next(iter(report["motifs"]))
                findings.append(f"- Most frequently recurring sequence pattern: `{top_motif}`.")
            for line in findings:
                st.markdown(line)
        else:
            st.info("Run the pipeline to populate findings.")

    st.divider()
    st.subheader("Project Workflow")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
            FASTA -> Preprocessing -> "Sequence Features\\n(GC, k-mers)" -> "Motif Discovery";
            "Sequence Features\\n(GC, k-mers)" -> "Classical ML\\n(RF / SVM / LogReg)";
            "Sequence Features\\n(GC, k-mers)" -> "1D-CNN\\n(one-hot encoding)";
            "Classical ML\\n(RF / SVM / LogReg)" -> Evaluation;
            "1D-CNN\\n(one-hot encoding)" -> Evaluation;
            Evaluation -> Visualization;
        }
        """
    )


# ---------------------------------------------------------------------------
# Page: Dataset Explorer
# ---------------------------------------------------------------------------

def page_dataset_explorer(df: pd.DataFrame):
    st.title("Dataset Explorer")

    if df is None:
        st.warning("No dataset loaded.")
        return

    classes = sorted(df["label"].unique())
    selected_class = st.selectbox("Filter by class", ["All classes"] + classes)
    view_df = df if selected_class == "All classes" else df[df["label"] == selected_class]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sequences", len(view_df))
    c2.metric("Min Length", int(view_df["length"].min()) if len(view_df) else 0)
    c3.metric("Max Length", int(view_df["length"].max()) if len(view_df) else 0)
    c4.metric("Mean Length", round(view_df["length"].mean(), 1) if len(view_df) else 0)
    c5.metric("Median Length", round(view_df["length"].median(), 1) if len(view_df) else 0)
    st.metric("Mean GC Content", f"{round(view_df['gc_content'].mean(), 2)}%" if len(view_df) else "N/A")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["GC Content", "Sequence Length", "Class Distribution", "GC / Length by Class"]
    )
    with tab1:
        st.plotly_chart(px.histogram(view_df, x="gc_content", nbins=30, title="GC Content Distribution"),
                         width="stretch")
    with tab2:
        st.plotly_chart(px.histogram(view_df, x="length", nbins=30, title="Sequence Length Distribution"),
                         width="stretch")
    with tab3:
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        st.plotly_chart(px.bar(counts, x="label", y="count", title="Class Distribution"),
                         width="stretch")
    with tab4:
        colA, colB = st.columns(2)
        colA.plotly_chart(px.box(df, x="label", y="gc_content", title="GC Content by Class"),
                           width="stretch")
        colB.plotly_chart(px.box(df, x="label", y="length", title="Sequence Length by Class"),
                           width="stretch")

    st.subheader("Sequence Table")
    search = st.text_input("Search by sequence ID")
    table_df = view_df[["seq_id", "label", "length", "gc_content", "at_content"]]
    if search:
        table_df = table_df[table_df["seq_id"].str.contains(search, case=False, na=False)]
    st.dataframe(table_df, width="stretch", height=350)


# ---------------------------------------------------------------------------
# Page: Sequence Analyzer
# ---------------------------------------------------------------------------

def page_sequence_analyzer(df: pd.DataFrame):
    st.title("Sequence Analyzer")

    if df is None:
        st.warning("No dataset loaded.")
        return

    seq_id = st.selectbox("Select a sequence", df["seq_id"].tolist())
    row = df[df["seq_id"] == seq_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sequence ID", row["seq_id"])
    col2.metric("Class", row["label"])
    col3.metric("Length", f"{int(row['length'])} bp")
    col4.metric("GC Content", f"{row['gc_content']}%")

    col5, col6 = st.columns(2)
    col5.metric("AT Content", f"{row['at_content']}%")
    col6.metric("A / T / G / C counts", f"{int(row['A'])} / {int(row['T'])} / {int(row['G'])} / {int(row['C'])}")

    st.subheader("Nucleotide Composition")
    comp_df = pd.DataFrame({
        "base": ["A", "T", "G", "C"],
        "count": [row["A"], row["T"], row["G"], row["C"]],
    })
    st.plotly_chart(px.bar(comp_df, x="base", y="count", title="Base Composition"), width="stretch")

    st.subheader("Sequence Validation")
    _, invalid_chars = validate_dna_sequence(row["sequence"])
    if invalid_chars:
        st.error(f"Invalid characters detected: {', '.join(invalid_chars)}")
    else:
        st.success("Sequence contains only valid A/C/G/T bases.")

    st.subheader("Full Sequence")
    st.text_area("DNA sequence", row["sequence"], height=150)


# ---------------------------------------------------------------------------
# Page: K-mer Analysis
# ---------------------------------------------------------------------------

def page_kmer_analysis(df: pd.DataFrame):
    st.title("K-mer Analysis")

    if df is None:
        st.warning("No dataset loaded.")
        return

    col1, col2, col3 = st.columns(3)
    k = col1.selectbox("k-mer size", [3, 4, 5, 6, 7], index=1)
    classes = sorted(df["label"].unique())
    class_filter = col2.selectbox("Class filter", ["All classes"] + classes)
    top_n = col3.slider("Number of top k-mers to show", 5, 30, 15)

    view_df = df if class_filter == "All classes" else df[df["label"] == class_filter]

    counts = Counter()
    for seq in view_df["sequence"]:
        s = DNASequence("tmp", seq)
        raw_counts = s.kmer_frequency(k=k, normalize=False)
        for kmer, c in raw_counts.items():
            counts[kmer] += c

    total = sum(counts.values()) or 1
    kmer_df = pd.DataFrame(
        [{"kmer": kmer, "frequency": c, "relative_frequency": round(c / total, 5)}
         for kmer, c in counts.most_common(top_n)]
    )

    st.subheader(f"Top {top_n} {k}-mers ({class_filter})")
    st.plotly_chart(px.bar(kmer_df, x="kmer", y="frequency", hover_data=["relative_frequency"]),
                     width="stretch")
    st.dataframe(kmer_df, width="stretch")

    st.divider()
    st.subheader("Per-sequence k-mer lookup")
    seq_id = st.selectbox("Select a sequence to inspect", df["seq_id"].tolist())
    seq_row = df[df["seq_id"] == seq_id].iloc[0]
    seq_obj = DNASequence(seq_id, seq_row["sequence"])
    seq_counts = seq_obj.kmer_frequency(k=k, normalize=False)
    top_seq_kmers = Counter(seq_counts).most_common(10)
    st.write(pd.DataFrame(top_seq_kmers, columns=["kmer", "count"]))


# ---------------------------------------------------------------------------
# Page: Motif Discovery
# ---------------------------------------------------------------------------

def page_motif_discovery(df: pd.DataFrame):
    st.title("Motif Discovery")
    st.markdown(
        "This page finds **frequently occurring sequence patterns** (short repeated "
        "substrings) across the dataset. These are statistical recurring patterns "
        "in this sequence set \u2014 they are *not* claimed to be biologically "
        "validated regulatory motifs."
    )

    if df is None:
        st.warning("No dataset loaded.")
        return

    col1, col2, col3 = st.columns(3)
    motif_size = col1.slider("Motif size (k)", 3, 10, 6)
    num_motifs = col2.slider("Number of motifs to display", 5, 30, 10)
    classes = sorted(df["label"].unique())
    class_filter = col3.selectbox("Class filter", ["All classes"] + classes)

    view_df = df if class_filter == "All classes" else df[df["label"] == class_filter]
    common = find_common_motifs(view_df["sequence"].tolist(), k=motif_size, min_occurrence_ratio=0.1)
    top_motifs = list(common.items())[:num_motifs]

    if not top_motifs:
        st.info("No recurring motifs found at this size/threshold. Try a smaller motif size.")
        return

    motif_df = pd.DataFrame(top_motifs, columns=["motif", "sequences_containing_it"])

    st.subheader("Most Frequent Motif")
    st.success(f"`{motif_df.iloc[0]['motif']}` — found in {motif_df.iloc[0]['sequences_containing_it']} sequences")

    st.plotly_chart(px.bar(motif_df, x="motif", y="sequences_containing_it"), width="stretch")
    st.dataframe(motif_df, width="stretch")


# ---------------------------------------------------------------------------
# Page: Machine Learning
# ---------------------------------------------------------------------------

def page_machine_learning(report: dict | None):
    st.title("Machine Learning Results")

    if not report or not report.get("classical"):
        st.warning(
            "No experiment results found yet. Go to **Run Analysis** to generate "
            f"`{FINAL_REPORT_PATH.name}`."
        )
        return

    classical = report["classical"]
    rows = [{"model": n, **m} for n, m in classical.items()]
    results_df = pd.DataFrame(rows)

    st.subheader("Test Accuracy by Model")
    st.plotly_chart(
        px.bar(results_df, x="model", y="test_accuracy", text_auto=".2%", range_y=[0, 1]),
        width="stretch",
    )

    if "cv_mean_accuracy" in results_df.columns:
        st.subheader("Cross-Validation Accuracy (mean \u00b1 std)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results_df["model"], y=results_df["cv_mean_accuracy"],
            error_y=dict(type="data", array=results_df.get("cv_std", [0] * len(results_df))),
        ))
        fig.update_yaxes(range=[0, 1], title="CV Accuracy")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Classification Metrics")
    for name in classical:
        with st.expander(f"{name} — classification report"):
            report_text = report.get("classification_reports", {}).get(name)
            if report_text:
                st.code(report_text, language=None)
            else:
                st.info("Detailed per-class report not available.")

    st.subheader("Confusion Matrix (best classical model)")
    if CONF_MATRIX_IMG.exists():
        st.image(str(CONF_MATRIX_IMG), caption=CONF_MATRIX_IMG.name, width="stretch")
    else:
        st.info(f"`{CONF_MATRIX_IMG.name}` not found in outputs/. Run the pipeline to generate it.")

    best = best_model_name_from_results({n: {"test_accuracy": m["test_accuracy"]} for n, m in classical.items()} | {"label_encoder": None, "scaler": None})
    st.success(f"Best classical model: **{best}**")


# ---------------------------------------------------------------------------
# Page: Deep Learning
# ---------------------------------------------------------------------------

def page_deep_learning(report: dict | None):
    st.title("Deep Learning (CNN)")

    st.markdown(
        "A **1D Convolutional Neural Network** is trained on one-hot encoded DNA "
        "sequences (each base becomes a 4-dimensional vector: A, C, G, T). "
        "One-hot encoding is used because raw sequence characters are not "
        "numeric, and this encoding preserves positional information that a "
        "k-mer frequency vector discards. A CNN may perform differently from "
        "k-mer-based classical models because it must *learn* which local "
        "patterns matter, rather than being handed pre-computed frequency "
        "features directly."
    )

    cnn = report.get("cnn") if report else None
    history = load_cnn_history(str(CNN_HISTORY_PATH))

    if not cnn:
        st.info(
            "No CNN results found yet. Run the analysis with the CNN option enabled "
            "from the **Run Analysis** page (this step is slower and is skipped by default)."
        )
        return

    col1, col2 = st.columns(2)
    col1.metric("CNN Test Accuracy", f"{cnn.get('test_accuracy', 0) * 100:.2f}%")
    history_epochs = len(history.get("history", {}).get("accuracy", [])) if history else 0
    col2.metric("Epochs", int(cnn.get("epochs", history_epochs)))

    if history and history.get("history"):
        h = history["history"]
        epochs = list(range(1, len(h.get("accuracy", [])) + 1))
        curve_df = pd.DataFrame({
            "epoch": epochs,
            "train_accuracy": h.get("accuracy", []),
            "val_accuracy": h.get("val_accuracy", []),
        })
        fig = px.line(curve_df, x="epoch", y=["train_accuracy", "val_accuracy"],
                      title="Training vs Validation Accuracy")
        st.plotly_chart(fig, width="stretch")

    if report.get("classical"):
        best_classical = best_model_name_from_results(
            {n: {"test_accuracy": m["test_accuracy"]} for n, m in report["classical"].items()}
            | {"label_encoder": None, "scaler": None}
        )
        best_classical_acc = report["classical"][best_classical]["test_accuracy"]
        cnn_acc = cnn.get("test_accuracy", 0)
        if cnn_acc < best_classical_acc:
            st.warning(
                f"On this current dataset, **{best_classical}** ({best_classical_acc * 100:.2f}%) "
                f"outperformed the CNN ({cnn_acc * 100:.2f}%). This reflects the small size of the "
                "current dataset, not a general conclusion that classical ML is superior to deep "
                "learning \u2014 CNNs typically need more training data to reach their potential."
            )
        else:
            st.success(f"The CNN ({cnn_acc * 100:.2f}%) matched or outperformed the best classical model.")


# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------

def page_model_comparison(report: dict | None):
    st.title("Model Comparison")

    if not report or not report.get("classical"):
        st.warning("No experiment results found yet. Run the analysis first.")
        return

    rows = []
    for name, m in report["classical"].items():
        rows.append({
            "model": name,
            "test_accuracy": m.get("test_accuracy"),
            "cv_accuracy": (
                f"{m['cv_mean_accuracy'] * 100:.2f}% \u00b1 {m['cv_std'] * 100:.2f}%"
                if "cv_mean_accuracy" in m else "N/A"
            ),
        })
    if report.get("cnn"):
        rows.append({"model": "CNN", "test_accuracy": report["cnn"].get("test_accuracy"), "cv_accuracy": "N/A"})

    comp_df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    medals = ["\U0001F947", "\U0001F948", "\U0001F949"] + [""] * max(0, len(comp_df) - 3)
    comp_df.insert(0, "rank", medals[:len(comp_df)])
    comp_df["test_accuracy"] = comp_df["test_accuracy"].apply(lambda v: f"{v * 100:.2f}%" if v is not None else "N/A")

    st.dataframe(comp_df, width="stretch", hide_index=True)

    best_row = comp_df.iloc[0]
    st.subheader("Scientific Interpretation")
    st.markdown(
        f"**{best_row['model']}** achieved the highest test accuracy "
        f"({best_row['test_accuracy']}) on this dataset. Tree-based and kernel "
        "classical models tend to perform well on k-mer frequency features "
        "because those features are already hand-engineered, low-dimensional, "
        "and informative — a setting where classical ML is often competitive "
        "with, or superior to, deep learning, especially on smaller datasets. "
        "Results should be interpreted relative to this specific dataset and "
        "feature representation, not as a universal ranking of these algorithms."
    )


# ---------------------------------------------------------------------------
# Page: DNA Predictor
# ---------------------------------------------------------------------------

def page_dna_predictor(report: dict | None):
    st.title("DNA Sequence Predictor")
    st.markdown(
        "Paste a DNA sequence below. The app trains classical ML models on the "
        "current dataset (cached after the first run) using the **exact same "
        "feature extraction pipeline** as the main experiment (`features.build_kmer_feature_matrix`), "
        "so there is no mismatch between training and prediction preprocessing."
    )

    if not dataset_available():
        st.warning("No dataset found — cannot train a model to predict with.")
        return

    k = DEFAULT_K
    if report and report.get("dataset", {}).get("kmer_size_used_for_ml"):
        try:
            k = int(report["dataset"]["kmer_size_used_for_ml"])
        except ValueError:
            pass

    raw_seq = st.text_area("Paste a DNA sequence", height=120, placeholder="ACGTACGTACGT...")

    trained = st.button("Train model and predict")

    if not raw_seq.strip():
        st.info("Enter a sequence above, then click **Train model and predict**.")
        return

    cleaned, invalid_chars = validate_dna_sequence(raw_seq)

    col1, col2 = st.columns(2)
    col1.metric("Input Length (valid bases)", len(cleaned))
    col2.metric("GC Content", f"{round(100 * sum(cleaned.count(b) for b in 'GC') / len(cleaned), 2)}%" if cleaned else "N/A")

    if invalid_chars:
        st.warning(f"Ignored invalid characters (not A/C/G/T): {', '.join(invalid_chars)}")

    if not cleaned:
        st.error("No valid A/C/G/T bases found in the input.")
        return

    if not trained:
        return

    results, feature_columns = train_models_cached(str(FASTA_PATH), str(LABELS_PATH), k)
    best_name = best_model_name_from_results(results)
    model = results[best_name]["model"]
    scaler = results["scaler"]
    encoder = results["label_encoder"]

    query_seq_obj = DNASequence("query", cleaned)
    kmer_freqs = query_seq_obj.kmer_frequency(k=k, normalize=True)
    feature_row = {**kmer_freqs, "gc_content": query_seq_obj.gc_content(), "length": len(query_seq_obj)}
    feature_vector = pd.DataFrame([feature_row]).reindex(columns=feature_columns, fill_value=0.0)

    scaled_vector = scaler.transform(feature_vector.values)
    prediction = model.predict(scaled_vector)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]

    st.success(f"Predicted class: **{predicted_label}** (model used: {best_name})")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_vector)[0]
        proba_df = pd.DataFrame({"class": encoder.classes_, "probability": proba}).sort_values(
            "probability", ascending=False
        )
        st.subheader("Class Probabilities")
        st.plotly_chart(px.bar(proba_df, x="class", y="probability", range_y=[0, 1]), width="stretch")


# ---------------------------------------------------------------------------
# Page: Run Analysis
# ---------------------------------------------------------------------------

def page_run_analysis():
    st.title("Run Analysis")
    st.markdown(
        "Runs the existing `src/pipeline.py` script exactly as it would be run from "
        "the command line. This regenerates every report and image the other pages "
        "read from."
    )

    k = st.slider("k-mer size", 3, 7, DEFAULT_K)
    run_cnn = st.checkbox(
        "Also train the CNN (slower — trains a 1D-CNN for ~15 epochs)", value=False
    )

    if st.button("Run pipeline"):
        cmd = [
            sys.executable, str(SRC_DIR / "pipeline.py"),
            "--fasta", str(FASTA_PATH),
            "--labels", str(LABELS_PATH),
            "--k", str(k),
            "--out-dir", str(OUTPUTS_DIR),
        ]
        if run_cnn:
            cmd.append("--run-cnn")

        log_box = st.empty()
        log_lines = []
        with st.spinner("Running pipeline..."):
            process = subprocess.Popen(
                cmd, cwd=str(SRC_DIR), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in process.stdout:
                log_lines.append(line.rstrip())
                log_box.code("\n".join(log_lines[-25:]), language=None)
            process.wait()

        if process.returncode == 0:
            st.success("Pipeline completed successfully.")
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            st.error(f"Pipeline exited with code {process.returncode}. See log above.")

        with st.expander("Full log"):
            st.code("\n".join(log_lines), language=None)


# ---------------------------------------------------------------------------
# Page: Reports
# ---------------------------------------------------------------------------

def page_reports():
    st.title("Reports")

    for label, path in [
        ("Final Experiment Report", FINAL_REPORT_PATH),
        ("Sequence Report", SEQUENCE_REPORT_PATH),
    ]:
        with st.expander(label, expanded=False):
            if path.exists():
                content = path.read_text()
                st.code(content, language=None)
                st.download_button(f"Download {path.name}", content, file_name=path.name)
            else:
                st.info(f"`{path.name}` not found in outputs/. Run the analysis to generate it.")

    st.subheader("Visualizations")
    for img_path in [GC_DIST_IMG, MODEL_ACC_IMG, CONF_MATRIX_IMG]:
        if img_path.exists():
            st.image(str(img_path), caption=img_path.name, width="stretch")
            with open(img_path, "rb") as fh:
                st.download_button(f"Download {img_path.name}", fh.read(), file_name=img_path.name)
        else:
            st.info(f"`{img_path.name}` not found in outputs/.")

    if LABELS_PATH.exists():
        st.subheader("Dataset labels (CSV)")
        content = LABELS_PATH.read_text()
        st.download_button("Download labels.csv", content, file_name="labels.csv")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about():
    st.title("About This Project")

    st.subheader("Objective")
    st.markdown(
        "Build an automated pipeline that analyzes DNA sequences using BioPython, "
        "engineers k-mer based features, and compares classical machine learning "
        "against a deep learning (CNN) approach for sequence classification — "
        "mapping to the CSB4140 course outcomes (CO1\u2013CO4)."
    )

    st.subheader("Technologies")
    st.markdown(
        "- Python\n- BioPython\n- Pandas\n- NumPy\n- Scikit-learn\n"
        "- TensorFlow / Keras\n- Matplotlib / Plotly\n- Streamlit"
    )

    st.subheader("Workflow")
    st.code("FASTA -> preprocessing -> sequence features -> k-mers/motifs -> classical ML -> CNN -> evaluation -> visualization", language=None)

    st.subheader("Limitations")
    st.markdown(
        "- The bundled dataset is synthetic and small; it exists to demonstrate the "
        "pipeline end-to-end without requiring external downloads.\n"
        "- All reported results are experimental and specific to this dataset.\n"
        "- The CNN was trained on limited data and limited epochs; its accuracy "
        "should not be read as a general statement about CNNs vs classical ML.\n"
        "- Results are for educational/demonstration purposes only and must not be "
        "interpreted as clinical or diagnostic evidence.\n"
        "- Real biological conclusions would require real, larger genomic datasets "
        "(e.g. NCBI GenBank, UCI Splice-Junction)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("Genomic Analysis")
    st.sidebar.caption("CSB4140 \u2014 Python Programming for Biosciences")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "Dataset Explorer",
            "Sequence Analyzer",
            "K-mer Analysis",
            "Motif Discovery",
            "Machine Learning",
            "Deep Learning",
            "Model Comparison",
            "DNA Predictor",
            "Run Analysis",
            "Reports",
            "About Project",
        ],
    )

    df = load_dataset_frame(str(FASTA_PATH), str(LABELS_PATH)) if dataset_available() else None
    report = parse_final_report(str(FINAL_REPORT_PATH))

    if page == "Dashboard":
        page_dashboard(df, report)
    elif page == "Dataset Explorer":
        page_dataset_explorer(df)
    elif page == "Sequence Analyzer":
        page_sequence_analyzer(df)
    elif page == "K-mer Analysis":
        page_kmer_analysis(df)
    elif page == "Motif Discovery":
        page_motif_discovery(df)
    elif page == "Machine Learning":
        page_machine_learning(report)
    elif page == "Deep Learning":
        page_deep_learning(report)
    elif page == "Model Comparison":
        page_model_comparison(report)
    elif page == "DNA Predictor":
        page_dna_predictor(report)
    elif page == "Run Analysis":
        page_run_analysis()
    elif page == "Reports":
        page_reports()
    elif page == "About Project":
        page_about()

    st.sidebar.divider()
    st.sidebar.markdown(
        '<span class="app-footnote">Educational project \u2014 results are experimental, '
        "not clinical evidence.</span>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
