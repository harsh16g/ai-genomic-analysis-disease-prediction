"""
sequences.py
------------
OOP layer for the Genomic Sequence Intelligence Engine (Module 1).

Design:
    GenomeSequence          -> base class (shared logic: cleaning, length, GC content)
        DNASequence         -> transcription, reverse complement, tetranucleotide frequency
        RNASequence         -> back-transcription, codon splitting
        ProteinSequence     -> amino acid composition, molecular weight (approx.)

This demonstrates CO1 (OOP, control structures, string handling) and
CO2 (BioPython-based sequence manipulation).
"""

from collections import Counter
from itertools import product
from Bio.Seq import Seq


class GenomeSequence:
    """Base class for any biological sequence."""

    VALID_CHARS = set("ACGTU N".replace(" ", ""))  # overridden by subclasses

    def __init__(self, seq_id: str, sequence: str, description: str = ""):
        self.seq_id = seq_id
        self.description = description
        self.raw_sequence = sequence.strip().upper()
        self.sequence = self._clean(self.raw_sequence)

    def _clean(self, sequence: str) -> str:
        """Remove whitespace/non-alphabetic noise. Subclasses may override
        to enforce an alphabet."""
        return "".join(ch for ch in sequence if ch.isalpha())

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        preview = self.sequence[:20] + ("..." if len(self.sequence) > 20 else "")
        return f"<{self.__class__.__name__} id={self.seq_id} len={len(self)} seq={preview}>"

    def gc_content(self) -> float:
        """Percentage of G/C bases. Meaningful for DNA/RNA; returns 0 for protein."""
        seq = self.sequence
        if not seq:
            return 0.0
        gc = sum(seq.count(b) for b in "GC")
        return round(100 * gc / len(seq), 2)

    def base_composition(self) -> dict:
        """Raw counts of each symbol in the sequence."""
        return dict(Counter(self.sequence))


class DNASequence(GenomeSequence):
    """Represents a DNA sequence. Adds transcription, reverse complement,
    and tetranucleotide (k-mer) frequency analysis used by BioPython unit
    of the syllabus."""

    VALID_CHARS = set("ACGT")

    def _clean(self, sequence: str) -> str:
        return "".join(ch for ch in sequence.upper() if ch in self.VALID_CHARS)

    def transcribe(self) -> "RNASequence":
        """DNA -> mRNA (T replaced by U)."""
        mrna = str(Seq(self.sequence).transcribe())
        return RNASequence(self.seq_id, mrna, self.description)

    def reverse_complement(self) -> str:
        return str(Seq(self.sequence).reverse_complement())

    def tetranucleotide_frequency(self) -> dict:
        """Frequency of every 4-mer in the sequence (the 'Tetra nucleotide
        Frequency: Counting Things' unit)."""
        return self.kmer_frequency(k=4)

    def kmer_frequency(self, k: int = 4, normalize: bool = True) -> dict:
        """General k-mer counter, used as the ML feature extractor.
        Every possible k-mer over {A,C,G,T} is included (zero-filled) so
        that feature vectors have a fixed, comparable length across
        different sequences.
        """
        all_kmers = ["".join(p) for p in product("ACGT", repeat=k)]
        counts = {kmer: 0 for kmer in all_kmers}
        seq = self.sequence
        total = max(len(seq) - k + 1, 1)
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in counts:
                counts[kmer] += 1
        if normalize:
            counts = {kmer: c / total for kmer, c in counts.items()}
        return counts

    def find_motif(self, motif: str) -> list:
        """Return all start indices (0-based) where `motif` occurs,
        including overlapping matches."""
        motif = motif.upper()
        seq = self.sequence
        positions = []
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
        return positions


class RNASequence(GenomeSequence):
    VALID_CHARS = set("ACGU")

    def _clean(self, sequence: str) -> str:
        return "".join(ch for ch in sequence.upper() if ch in self.VALID_CHARS)

    def back_transcribe(self) -> "DNASequence":
        dna = str(Seq(self.sequence).back_transcribe())
        return DNASequence(self.seq_id, dna, self.description)

    def codons(self) -> list:
        """Split into non-overlapping triplets (drops an incomplete tail)."""
        seq = self.sequence
        return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


class ProteinSequence(GenomeSequence):
    VALID_CHARS = set("ACDEFGHIKLMNPQRSTVWY")

    # Very rough average residue masses (Da) for an approximate MW estimate.
    _AVG_RESIDUE_MASS = 110.0

    def _clean(self, sequence: str) -> str:
        return "".join(ch for ch in sequence.upper() if ch in self.VALID_CHARS)

    def amino_acid_composition(self) -> dict:
        seq = self.sequence
        if not seq:
            return {}
        counts = Counter(seq)
        return {aa: round(100 * c / len(seq), 2) for aa, c in counts.items()}

    def approx_molecular_weight(self) -> float:
        """Rough estimate only (average residue mass minus water per bond
        is ignored for simplicity) — enough for a demo-level report."""
        return round(len(self.sequence) * self._AVG_RESIDUE_MASS, 1)


def load_fasta(path: str, molecule: str = "dna") -> list:
    """Parse a FASTA file into a list of GenomeSequence subclass instances.

    molecule: "dna" | "rna" | "protein"
    """
    from Bio import SeqIO

    cls_map = {"dna": DNASequence, "rna": RNASequence, "protein": ProteinSequence}
    cls = cls_map.get(molecule.lower())
    if cls is None:
        raise ValueError(f"Unknown molecule type: {molecule}")

    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append(cls(record.id, str(record.seq), record.description))
    return records
