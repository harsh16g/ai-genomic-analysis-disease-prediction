"""
alignment_motifs.py
--------------------
Sequence similarity (pairwise alignment) and motif discovery — the CO3
"Analyse sequence manipulation, alignment and basic comparative analysis"
outcome.
"""

from Bio import Align


def pairwise_similarity(seq1: str, seq2: str) -> dict:
    """Global pairwise alignment score + percent identity using
    BioPython's PairwiseAligner (replacement for the deprecated
    Bio.pairwise2)."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(seq1, seq2)[0]
    aligned_a, aligned_b = str(alignment[0]), str(alignment[1])

    matches = sum(1 for a, b in zip(aligned_a, aligned_b) if a == b and a != "-")
    length = max(len(seq1), len(seq2))
    identity = round(100 * matches / length, 2) if length else 0.0

    return {
        "score": alignment.score,
        "percent_identity": identity,
        "alignment": str(alignment),
    }


def find_common_motifs(sequences: list, k: int = 6, min_occurrence_ratio: float = 0.5) -> dict:
    """
    Naive motif discovery: find k-length substrings that recur across at
    least `min_occurrence_ratio` fraction of the given sequences.

    sequences: list[str] (raw sequence strings)
    Returns {motif: count_of_sequences_containing_it}
    """
    from collections import defaultdict

    motif_seq_count = defaultdict(int)
    for seq in sequences:
        seen_in_this_seq = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            seen_in_this_seq.add(kmer)
        for kmer in seen_in_this_seq:
            motif_seq_count[kmer] += 1

    threshold = max(1, int(len(sequences) * min_occurrence_ratio))
    common = {m: c for m, c in motif_seq_count.items() if c >= threshold}
    return dict(sorted(common.items(), key=lambda x: -x[1]))
