"""
features.py
------------
Turns a list of DNASequence objects into a numeric feature matrix (Pandas
DataFrame) suitable for classical ML and deep learning models.

This is the CO2 -> CO4 bridge: BioPython-derived features feeding a
Pandas/NumPy-based ML pipeline.
"""

from itertools import product
import numpy as np
import pandas as pd

from sequences import DNASequence


def build_kmer_feature_matrix(sequences: list, labels: list, k: int = 4) -> pd.DataFrame:
    """
    sequences: list[DNASequence]
    labels:    list[str] (same length as sequences) — e.g. organism / class name

    Returns a DataFrame: one row per sequence, one column per possible
    k-mer, plus 'gc_content', 'length', and 'label' columns.
    """
    assert len(sequences) == len(labels), "sequences and labels must align"

    all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
    rows = []
    for seq_obj, label in zip(sequences, labels):
        freqs = seq_obj.kmer_frequency(k=k, normalize=True)
        row = {kmer: freqs[kmer] for kmer in all_kmers}
        row["gc_content"] = seq_obj.gc_content()
        row["length"] = len(seq_obj)
        row["label"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def one_hot_encode_sequences(sequences: list, max_len: int = None) -> np.ndarray:
    """
    Encode raw DNA sequences as one-hot arrays for the CNN (Module 3).
    Shape returned: (n_samples, max_len, 4)  channels = A,C,G,T
    Sequences shorter than max_len are zero-padded; longer ones truncated.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    encoded = np.zeros((len(sequences), max_len, 4), dtype=np.float32)
    for i, seq_obj in enumerate(sequences):
        seq = seq_obj.sequence[:max_len]
        for j, base in enumerate(seq):
            idx = mapping.get(base)
            if idx is not None:
                encoded[i, j, idx] = 1.0
    return encoded
