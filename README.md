# Multi-Level Genomic Sequence Analysis & Classification System

A Python-based bioinformatics and machine learning project for analyzing DNA sequences using **BioPython, sequence analysis, classical machine learning, and deep learning**.

The project implements an automated end-to-end workflow that starts with FASTA sequences and produces sequence statistics, motif analysis, machine-learning predictions, visualizations, and model-comparison results.

## Project Overview

The system is organized into four modules:

```text
FASTA Sequences
      |
      v
+-------------------------------+
| Module 1: Sequence Analysis   |
| BioPython + OOP               |
+-------------------------------+
      |
      v
+-------------------------------+
| Module 2: Classical ML        |
| k-mers + RF / SVM / LR        |
+-------------------------------+
      |
      v
+-------------------------------+
| Module 3: Deep Learning       |
| One-hot Encoding + 1D-CNN     |
+-------------------------------+
      |
      v
+-------------------------------+
| Module 4: Automation          |
| End-to-End Pipeline           |
+-------------------------------+
      |
      v
Reports + Visualizations + Results
```

## Main Features

### Module 1 — Genomic Sequence Analysis

Implemented in:

* `src/sequences.py`
* `src/alignment_motifs.py`

Features include:

* Object-oriented genomic sequence hierarchy
* `GenomeSequence` base class
* `DNASequence`, `RNASequence`, and `ProteinSequence`
* DNA transcription to RNA
* Reverse complement
* GC-content calculation
* k-mer and tetranucleotide frequency analysis
* Motif searching
* Pairwise sequence alignment
* Percent identity calculation
* Cross-dataset motif discovery

### Module 2 — Classical Machine Learning

Implemented in:

* `src/features.py`
* `src/classical_ml.py`

The DNA sequences are converted into numerical features using **k-mer frequency representations**.

Three classical classifiers are evaluated:

* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression

The workflow includes:

* Feature-matrix construction
* Train/test splitting
* Model training
* Test-set evaluation
* Cross-validation
* Precision, recall, and F1-score
* Confusion matrix generation
* Model comparison

### Module 3 — Deep Learning

Implemented in:

* `src/deep_learning.py`

The sequences are converted into one-hot encoded representations and passed to a **1D Convolutional Neural Network (CNN)** using TensorFlow/Keras.

The CNN provides a direct deep-learning comparison against the classical machine-learning models.

### Module 4 — Automated Pipeline

Implemented in:

* `src/pipeline.py`

The pipeline automates the complete workflow using a single command:

```bash
python src/pipeline.py
```

It performs:

1. FASTA loading
2. Sequence analysis
3. Per-sequence report generation
4. Motif discovery
5. k-mer feature extraction
6. Classical ML training and evaluation
7. Visualization generation
8. Optional CNN training

This demonstrates automation of a biological data-analysis workflow.

---

## Dataset

The included demonstration dataset contains **240 synthetic DNA sequences** belonging to four classes:

* `OrganismA_lowGC`
* `OrganismB_midGC`
* `OrganismC_highGC`
* `OrganismD_midGC_alt`

The synthetic dataset allows the entire project to run without external data downloads.

The project can later be adapted to real genomic datasets by providing a compatible FASTA file and label CSV.

### Input format

FASTA:

```text
>sequence_id
ACGTACGTACGT...
```

Labels CSV:

```text
seq_id,label
sequence_id,OrganismA_lowGC
```

---

## Verified Experimental Results

The complete pipeline was executed successfully on the included 240-sequence dataset.

### Classical ML results

| Model               | Test Accuracy | Cross-Validation Accuracy |
| ------------------- | ------------: | ------------------------: |
| **Random Forest**   |    **93.33%** |        **91.67% ± 3.51%** |
| Logistic Regression |        86.67% |            86.11% ± 5.27% |
| SVM                 |        85.00% |            86.67% ± 4.08% |

### CNN result

The 1D CNN was trained for 15 epochs using one-hot encoded DNA sequences.

**CNN test accuracy: 75.00%**

### Main finding

```text
Random Forest       93.33%
Logistic Regression 86.67%
SVM                 85.00%
CNN                 75.00%
```

Random Forest produced the best test accuracy on this small synthetic dataset.

The lower CNN performance is a useful experimental observation rather than a failure. Deep neural networks generally benefit from larger training datasets, while k-mer-based representations combined with classical models can provide strong baselines when the available dataset is relatively small.

This experiment therefore demonstrates why model selection should consider both the algorithm and the amount and nature of available training data.

---

## Generated Outputs

After running the pipeline, the `outputs/` directory contains generated analysis results.

### `sequence_report.txt`

Contains per-sequence:

* Sequence ID
* Sequence length
* GC content
* Top tetranucleotide frequencies

### `gc_distribution.png`

Visualizes GC-content distributions across the four sequence classes.

### `model_accuracy_comparison.png`

Shows the test accuracy comparison of the classical machine-learning models.

### `confusion_matrix_RandomForest.png`

Shows the Random Forest classification performance across the four classes.

---

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

For the CNN comparison, install TensorFlow:

```powershell
python -m pip install tensorflow
```

Verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Running the Project

From the project root:

```powershell
python src\pipeline.py
```

This runs the sequence-analysis and classical-ML workflow.

To include the CNN:

```powershell
python src\pipeline.py --run-cnn
```

The default paths are:

```text
data/sample_sequences.fasta
data/labels.csv
outputs/
```

A different dataset can be supplied using:

```powershell
python src\pipeline.py `
    --fasta path\to\sequences.fasta `
    --labels path\to\labels.csv `
    --k 4 `
    --run-cnn `
    --out-dir outputs
```

---

## Project Structure

```text
genomic_analysis_project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sample_sequences.fasta
│   └── labels.csv
│
├── outputs/
│   ├── sequence_report.txt
│   ├── gc_distribution.png
│   ├── model_accuracy_comparison.png
│   └── confusion_matrix_RandomForest.png
│
└── src/
    ├── sequences.py
    ├── alignment_motifs.py
    ├── features.py
    ├── data_generator.py
    ├── classical_ml.py
    ├── deep_learning.py
    ├── visualize.py
    └── pipeline.py
```

---

## Course Outcome Mapping

| Course Outcome                                 | Project Implementation                            |
| ---------------------------------------------- |---------------------------------------------------|
| **CO1** — Python programming and OOP           | Sequence class hierarchy, inheritance, functions, control structures |
| **CO2** — Python libraries and biological data | BioPython, NumPy, Pandas, Matplotlib and Seaborn  |
| **CO3** — Biological sequence analysis         | GC analysis, k-mers, transcription, reverse complement, alignment and motif discovery |
| **CO4** — Automation of biological workflows   | `pipeline.py` integrates the complete analysis into a single automated workflow       |


## Technologies Used

* Python 3
* BioPython
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* TensorFlow / Keras

---

## Limitations

The included dataset is **synthetic and relatively small**, so the reported model accuracies should not be interpreted as biological performance on real-world genomic datasets.

The project is primarily designed to demonstrate:

* Bioinformatics programming
* Feature engineering
* Machine-learning classification
* Deep-learning comparison
* Visualization
* Workflow automation

For a research-oriented application, the pipeline should be evaluated using larger, biologically validated datasets and appropriate train/test separation strategies.

---

## Possible Future Extensions

* Replace the synthetic dataset with NCBI or other validated genomic datasets
* Add additional sequence features
* Hyperparameter tuning using GridSearchCV
* Compare additional machine-learning algorithms
* Add ROC-AUC and precision-recall curves
* Improve CNN architecture and training strategy
* Add explainable-AI methods
* Add a Streamlit web interface
* Add database-backed sequence storage
* Deploy the project as an interactive web application

---

## Author

**Genomic Sequence Analysis & Classification Project**

Developed as an academic Python/Bioinformatics project demonstrating AI-assisted genomic data analysis.
