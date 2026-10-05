"""
data_generator.py
------------------
Generates a synthetic multi-class DNA dataset and writes it to FASTA.

Why synthetic data:
    Real datasets (NCBI GenBank, UCI Splice-Junction, Kaggle) are the
    recommended source for the actual submission — swap them in via
    `load_fasta()` in sequences.py. This generator lets the full
    pipeline (parsing -> features -> classical ML -> CNN -> report) run
    and be demoed offline, with each synthetic "organism class" biased
    toward a different GC content and a distinct motif, so the models
    have real signal to learn from.

Run directly:
    python data_generator.py
"""

import random

random.seed(42)

CLASSES = {
    # class_name: (gc_bias 0-1, signature_motif)
    "OrganismA_lowGC": (0.30, "TTAAGG"),
    "OrganismB_midGC": (0.50, "GCTAGC"),
    "OrganismC_highGC": (0.70, "GGCCGG"),
    "OrganismD_midGC_alt": (0.50, "ATCGAT"),
}

BASES = "ACGT"


def _weighted_base(gc_bias: float) -> str:
    if random.random() < gc_bias:
        return random.choice("GC")
    return random.choice("AT")


def generate_sequence(length: int, gc_bias: float, motif: str) -> str:
    seq = [_weighted_base(gc_bias) for _ in range(length)]
    # Insert the class-signature motif at a random position so classifiers
    # have a learnable pattern beyond raw GC content.
    insert_at = random.randint(0, max(length - len(motif), 0))
    seq[insert_at:insert_at + len(motif)] = list(motif)
    return "".join(seq)


def generate_dataset(n_per_class: int = 60, min_len: int = 150, max_len: int = 300) -> list:
    """Returns list of (seq_id, sequence, label, description) tuples."""
    records = []
    counter = 0
    for label, (gc_bias, motif) in CLASSES.items():
        for _ in range(n_per_class):
            length = random.randint(min_len, max_len)
            seq = generate_sequence(length, gc_bias, motif)
            counter += 1
            seq_id = f"{label}_{counter:04d}"
            records.append((seq_id, seq, label, f"synthetic|{label}"))
    random.shuffle(records)
    return records


def write_fasta(records: list, path: str, labels_path: str = None):
    with open(path, "w") as fh:
        for seq_id, seq, label, desc in records:
            fh.write(f">{seq_id} {desc}\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i:i + 70] + "\n")

    if labels_path:
        with open(labels_path, "w") as fh:
            fh.write("seq_id,label\n")
            for seq_id, _, label, _ in records:
                fh.write(f"{seq_id},{label}\n")


if __name__ == "__main__":
    recs = generate_dataset()
    write_fasta(recs, "../data/sample_sequences.fasta", "../data/labels.csv")
    print(f"Wrote {len(recs)} sequences across {len(CLASSES)} classes to data/")
