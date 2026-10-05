
"""
pipeline.py
-----------
Module 4: Automated genomic sequence analysis and classification pipeline.

Workflow:

    FASTA file
        |
        v
    Sequence analysis
        |
        v
    Per-sequence report
        |
        v
    Motif discovery
        |
        v
    k-mer feature extraction
        |
        v
    Classical ML comparison
        |
        +---- Random Forest
        +---- SVM
        +---- Logistic Regression
        |
        v
    Optional 1D CNN comparison
        |
        v
    Visualizations + final experiment report

Run from the project root:

    python src\\pipeline.py

Run with CNN:

    python src\\pipeline.py --run-cnn

Custom dataset:

    python src\\pipeline.py ^
        --fasta data\\sample_sequences.fasta ^
        --labels data\\labels.csv ^
        --k 4 ^
        --run-cnn ^
        --out-dir outputs
"""

import argparse
import csv
import json
import os

from sequences import load_fasta
from features import build_kmer_feature_matrix, one_hot_encode_sequences
from classical_ml import train_and_compare, print_comparison
from alignment_motifs import find_common_motifs
from visualize import (
    plot_confusion_matrix,
    plot_gc_distribution,
    plot_model_accuracy_comparison,
)


def load_labels(labels_csv: str) -> dict:
    """
    Load sequence labels from a CSV file.

    Expected CSV format:

        seq_id,label

    Returns:
        Dictionary mapping sequence IDs to class labels.
    """
    mapping = {}

    with open(labels_csv, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            mapping[row["seq_id"]] = row["label"]

    return mapping


def per_sequence_report(sequences: list, out_path: str):
    """
    Generate a text report containing basic statistics
    for every genomic sequence.
    """

    with open(out_path, "w", encoding="utf-8") as fh:

        fh.write("PER-SEQUENCE GENOMIC ANALYSIS REPORT\n")
        fh.write("=" * 55 + "\n\n")

        for s in sequences:

            fh.write(f"ID: {s.seq_id}\n")
            fh.write(f"  Length: {len(s)}\n")
            fh.write(f"  GC content: {s.gc_content()}%\n")

            top_tetranucleotides = sorted(
                s.tetranucleotide_frequency().items(),
                key=lambda x: -x[1],
            )[:5]

            fh.write(
                f"  Tetranucleotide (top 5): "
                f"{top_tetranucleotides}\n\n"
            )

    print(f"Per-sequence report written to {out_path}")


def write_final_report(
    sequences: list,
    labels: list,
    top_motifs: list,
    results: dict,
    out_path: str,
    cnn_accuracy: float = None,
):
    """
    Write a final summary report containing:

    - Dataset information
    - Recurring motifs
    - Classical ML performance
    - CNN performance, if enabled
    - Model ranking
    - Final interpretation
    """

    with open(out_path, "w", encoding="utf-8") as fh:

        fh.write(
            "GENOMIC SEQUENCE ANALYSIS & CLASSIFICATION REPORT\n"
        )
        fh.write("=" * 60 + "\n\n")

        # ---------------------------------------------------------
        # Dataset summary
        # ---------------------------------------------------------

        fh.write("1. DATASET SUMMARY\n")
        fh.write("-" * 60 + "\n")

        fh.write(
            f"Number of sequences: {len(sequences)}\n"
        )

        fh.write(
            f"Number of classes: {len(set(labels))}\n"
        )

        fh.write(
            f"Classes: {', '.join(sorted(set(labels)))}\n\n"
        )

        # ---------------------------------------------------------
        # Motif analysis
        # ---------------------------------------------------------

        fh.write("2. TOP RECURRING 6-MERS\n")
        fh.write("-" * 60 + "\n")

        for motif, count in top_motifs:
            fh.write(
                f"{motif}: found in {count} sequences\n"
            )

        fh.write("\n")

        # ---------------------------------------------------------
        # Classical ML results
        # ---------------------------------------------------------

        fh.write("3. CLASSICAL MACHINE LEARNING RESULTS\n")
        fh.write("-" * 60 + "\n")

        model_accuracies = {}

        for name, result in results.items():

            if name in ("label_encoder", "scaler"):
                continue

            accuracy = result["test_accuracy"]

            model_accuracies[name] = accuracy

            fh.write(f"\n{name}\n")

            fh.write(
                f"Test accuracy: "
                f"{accuracy:.4f} "
                f"({accuracy * 100:.2f}%)\n"
            )

            fh.write(
                f"Cross-validation accuracy: "
                f"{result['cv_mean_accuracy']:.4f} "
                f"+/- {result['cv_std']:.4f}\n"
            )

            fh.write("\nClassification report:\n")

            fh.write(
                result["classification_report"]
            )

            fh.write("\n")

        # ---------------------------------------------------------
        # CNN result
        # ---------------------------------------------------------

        if cnn_accuracy is not None:

            model_accuracies["CNN"] = cnn_accuracy

            fh.write(
                "\n4. DEEP LEARNING RESULT\n"
            )

            fh.write("-" * 60 + "\n")

            fh.write(
                "Model: 1D Convolutional Neural Network\n"
            )

            fh.write(
                f"Test accuracy: "
                f"{cnn_accuracy:.4f} "
                f"({cnn_accuracy * 100:.2f}%)\n"
            )

            fh.write(
                "\nThe CNN was trained using one-hot encoded "
                "DNA sequences.\n"
            )

        # ---------------------------------------------------------
        # Final comparison
        # ---------------------------------------------------------

        if model_accuracies:

            best_model = max(
                model_accuracies,
                key=model_accuracies.get,
            )

            best_accuracy = model_accuracies[best_model]

            fh.write(
                "\n5. FINAL MODEL COMPARISON\n"
            )

            fh.write("-" * 60 + "\n")

            for name, accuracy in sorted(
                model_accuracies.items(),
                key=lambda x: x[1],
                reverse=True,
            ):

                fh.write(
                    f"{name:<22} "
                    f"{accuracy * 100:.2f}%\n"
                )

            fh.write("\n")

            fh.write(
                f"Best model: {best_model}\n"
            )

            fh.write(
                f"Best test accuracy: "
                f"{best_accuracy * 100:.2f}%\n"
            )

            # -----------------------------------------------------
            # Interpretation
            # -----------------------------------------------------

            fh.write(
                "\n6. INTERPRETATION\n"
            )

            fh.write("-" * 60 + "\n")

            fh.write(
                "The experiment compares classical machine-learning "
                "methods using k-mer frequency features with a "
                "1D convolutional neural network using one-hot "
                "encoded DNA sequences.\n\n"
            )

            if (
                "RandomForest" in model_accuracies
                and "CNN" in model_accuracies
            ):

                rf_accuracy = model_accuracies["RandomForest"]

                if rf_accuracy > cnn_accuracy:

                    fh.write(
                        "Random Forest achieved higher test accuracy "
                        "than the CNN on this dataset. This suggests "
                        "that the k-mer feature representation combined "
                        "with a classical ensemble model is effective "
                        "for this relatively small synthetic dataset.\n\n"
                    )

                    fh.write(
                        "The lower CNN performance should not be "
                        "interpreted as evidence that CNNs are inferior "
                        "for genomic sequence analysis in general. "
                        "Deep-learning models typically benefit from "
                        "larger datasets and more extensive training.\n"
                    )

                elif cnn_accuracy > rf_accuracy:

                    fh.write(
                        "The CNN achieved higher test accuracy than "
                        "Random Forest on this dataset, suggesting "
                        "that the learned sequence representation "
                        "captured useful patterns for classification.\n"
                    )

                else:

                    fh.write(
                        "Random Forest and CNN achieved the same "
                        "test accuracy on this dataset.\n"
                    )

            fh.write(
                "\nNote: The dataset used in this project is synthetic "
                "and relatively small. Therefore, the reported "
                "accuracies demonstrate the behavior of the implemented "
                "methods rather than real-world biological prediction "
                "performance.\n"
            )

    print(
        f"Final experiment report written to {out_path}"
    )


def run_pipeline(
    fasta_path: str,
    labels_path: str,
    k: int,
    run_cnn: bool,
    out_dir: str,
):
    """
    Execute the complete genomic analysis pipeline.
    """

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load sequences
    # -------------------------------------------------------------

    print(
        f"[1/5] Loading sequences from {fasta_path} ..."
    )

    sequences = load_fasta(
        fasta_path,
        molecule="dna",
    )

    print(
        f"  Loaded {len(sequences)} sequences."
    )

    # -------------------------------------------------------------
    # Load labels
    # -------------------------------------------------------------

    label_map = load_labels(labels_path)

    labels = [
        label_map[s.seq_id]
        for s in sequences
    ]

    # -------------------------------------------------------------
    # Step 2: Per-sequence report
    # -------------------------------------------------------------

    print(
        "[2/5] Writing per-sequence BioPython report ..."
    )

    per_sequence_report(
        sequences,
        os.path.join(
            out_dir,
            "sequence_report.txt",
        ),
    )

    # -------------------------------------------------------------
    # Step 3: Motif discovery
    # -------------------------------------------------------------

    print(
        "[3/5] Discovering common motifs across the dataset ..."
    )

    common_motifs = find_common_motifs(
        [s.sequence for s in sequences],
        k=6,
        min_occurrence_ratio=0.15,
    )

    top_motifs = list(
        common_motifs.items()
    )[:10]

    print(
        f"  Top recurring 6-mers: {top_motifs}"
    )

    # -------------------------------------------------------------
    # Step 4: Feature extraction + classical ML
    # -------------------------------------------------------------

    print(
        f"[4/5] Building k-mer (k={k}) feature matrix "
        "and training classical ML models ..."
    )

    df = build_kmer_feature_matrix(
        sequences,
        labels,
        k=k,
    )

    results = train_and_compare(df)

    print_comparison(results)

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------

    print(
        "      Saving visualizations ..."
    )

    plot_gc_distribution(
        sequences,
        labels,
        os.path.join(
            out_dir,
            "gc_distribution.png",
        ),
    )

    plot_model_accuracy_comparison(
        results,
        os.path.join(
            out_dir,
            "model_accuracy_comparison.png",
        ),
    )

    # Best classical model
    best_model_name = max(
        (
            name
            for name in results
            if name not in ("label_encoder", "scaler")
        ),
        key=lambda name:
        results[name]["test_accuracy"],
    )

    plot_confusion_matrix(
        results[best_model_name]["confusion_matrix"],
        results["label_encoder"].classes_,
        os.path.join(
            out_dir,
            f"confusion_matrix_{best_model_name}.png",
        ),
        title=f"Confusion Matrix - {best_model_name}",
    )

    # -------------------------------------------------------------
    # Step 5: Optional CNN
    # -------------------------------------------------------------

    cnn_accuracy = None

    if run_cnn:

        print(
            "[5/5] Training CNN on one-hot encoded "
            "sequences for comparison ..."
        )

        try:

            from deep_learning import train_cnn
            from sklearn.model_selection import train_test_split

            X = one_hot_encode_sequences(
                sequences
            )

            y = results[
                "label_encoder"
            ].transform(labels)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )

            cnn_result = train_cnn(
                X_train,
                y_train,
                X_test,
                y_test,
                n_classes=len(set(y)),
            )

            with open(
                os.path.join(out_dir, "cnn_history.json"),
                "w",
                encoding="utf-8",
            ) as history_file:
                json.dump(cnn_result["history"], history_file, indent=2)

            cnn_accuracy = cnn_result[
                "test_accuracy"
            ]

            print(
                f"  CNN test accuracy: "
                f"{cnn_accuracy:.4f} "
                f"({cnn_accuracy * 100:.2f}%)"
            )

        except ImportError as error:

            print(
                f"  Skipping CNN step: {error}"
            )

            print(
                "  Install TensorFlow with:"
            )

            print(
                "  python -m pip install tensorflow"
            )

    else:

        print(
            "[5/5] Skipped CNN step "
            "(pass --run-cnn to enable)."
        )

    # -------------------------------------------------------------
    # Final experiment report
    # -------------------------------------------------------------

    write_final_report(
        sequences=sequences,
        labels=labels,
        top_motifs=top_motifs,
        results=results,
        out_path=os.path.join(
            out_dir,
            "final_experiment_report.txt",
        ),
        cnn_accuracy=cnn_accuracy,
    )

    # -------------------------------------------------------------
    # Completion message
    # -------------------------------------------------------------

    print(
        f"\nDone. Reports written to: {out_dir}/"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Genomic Sequence Analysis "
            "& Classification Pipeline"
        )
    )

    # IMPORTANT:
    # These paths are relative to the PROJECT ROOT,
    # where the user runs: python src\\pipeline.py

    parser.add_argument(
        "--fasta",
        default="data/sample_sequences.fasta",
        help="Path to the input FASTA file",
    )

    parser.add_argument(
        "--labels",
        default="data/labels.csv",
        help="Path to the labels CSV file",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="k-mer size for feature extraction",
    )

    parser.add_argument(
        "--run-cnn",
        action="store_true",
        help="Also train the CNN comparison model",
    )

    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory where reports and plots are saved",
    )

    args = parser.parse_args()

    run_pipeline(
        args.fasta,
        args.labels,
        args.k,
        args.run_cnn,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
    