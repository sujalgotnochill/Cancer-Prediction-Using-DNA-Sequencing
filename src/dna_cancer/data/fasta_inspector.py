"""Streaming inspection utility for large multi-file FASTA genomic collections."""

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def format_bytes(size: float) -> str:
    """Format byte size into human-readable unit (B, KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def detect_file_encoding(file_path: Union[str, Path], probe_bytes: int = 8192) -> str:
    """Detect file encoding from initial bytes with UTF-8 priority and latin-1 fallback."""
    path = Path(file_path)
    with open(path, "rb") as fp:
        raw = fp.read(probe_bytes)
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


@dataclass
class FastaFileSummary:
    """Detailed forensic summary of an individual FASTA file."""
    filename: str
    extension: str
    size_bytes: int
    size_human: str
    encoding: str
    is_valid_fasta: bool
    first_5_headers: List[str]
    first_5_previews: List[str]
    total_records: int
    total_nucleotides: int
    min_length: int
    max_length: int
    avg_length: float
    alphabet_distribution: Dict[str, int]
    alphabet_percentages: Dict[str, float]
    internal_duplicates: int
    unique_sequences: int
    accession_prefixes: Dict[str, int]
    sample_genes: List[str]
    sample_proteins: List[str]
    sample_locus_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_fasta_file(
    file_path: Union[str, Path],
    max_hash_records: Optional[int] = None,
    progress_interval: int = 500000,
) -> Tuple[FastaFileSummary, Set[bytes]]:
    """Stream-process a single FASTA file of any size without loading entire sequences into RAM.

    Args:
        file_path: Path to target FASTA file.
        max_hash_records: Optional upper limit on unique hash collection (for memory throttling).
        progress_interval: Number of lines between logger progress updates.

    Returns:
        Tuple of (FastaFileSummary, set_of_8byte_sequence_hashes).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    file_size = path.stat().st_size
    human_size = format_bytes(file_size)
    encoding = detect_file_encoding(path)

    total_records = 0
    total_nucs = 0
    min_len = float("inf")
    max_len = 0
    first_5_headers: List[str] = []
    first_5_previews: List[str] = []
    base_counts: Dict[str, int] = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0}
    other_bases: Dict[str, int] = {}

    accession_prefixes: Dict[str, int] = {}
    genes_found: List[str] = []
    proteins_found: List[str] = []
    locus_tags_found: List[str] = []

    seq_hashes: Set[bytes] = set()
    internal_duplicates = 0
    is_valid_fasta = True

    current_header: Optional[str] = None
    current_seq_chunks: List[str] = []

    def process_record(header: str, seq_str: str) -> None:
        nonlocal total_records, total_nucs, min_len, max_len, internal_duplicates, is_valid_fasta
        if not seq_str:
            is_valid_fasta = False
            return

        total_records += 1
        seq_len = len(seq_str)
        total_nucs += seq_len
        if seq_len < min_len:
            min_len = seq_len
        if seq_len > max_len:
            max_len = seq_len

        if len(first_5_headers) < 5:
            first_5_headers.append(header)
            first_5_previews.append(seq_str[:80])

        # C-level string counting for fast execution
        a = seq_str.count("A")
        c = seq_str.count("C")
        g = seq_str.count("G")
        t = seq_str.count("T")
        n = seq_str.count("N")
        base_counts["A"] += a
        base_counts["C"] += c
        base_counts["G"] += g
        base_counts["T"] += t
        base_counts["N"] += n

        diff = seq_len - (a + c + g + t + n)
        if diff > 0:
            for ch in seq_str:
                if ch not in ("A", "C", "G", "T", "N"):
                    other_bases[ch] = other_bases.get(ch, 0) + 1

        # Truncated 8-byte SHA-256 hash for memory-efficient duplicate tracking
        if max_hash_records is None or len(seq_hashes) < max_hash_records:
            h = hashlib.sha256(seq_str.encode("ascii", errors="replace")).digest()[:8]
            if h in seq_hashes:
                internal_duplicates += 1
            else:
                seq_hashes.add(h)

        # Header metadata parsing
        acc_match = re.search(r">lcl\|([^._]+)", header)
        if acc_match:
            pfx = acc_match.group(1)
            accession_prefixes[pfx] = accession_prefixes.get(pfx, 0) + 1

        if len(genes_found) < 20:
            g_match = re.search(r"\[gene=([^\]]+)\]", header)
            if g_match and g_match.group(1) not in genes_found:
                genes_found.append(g_match.group(1))

        if len(proteins_found) < 20:
            p_match = re.search(r"\[protein=([^\]]+)\]", header)
            if p_match and p_match.group(1) not in proteins_found:
                proteins_found.append(p_match.group(1))

        if len(locus_tags_found) < 20:
            l_match = re.search(r"\[locus_tag=([^\]]+)\]", header)
            if l_match and l_match.group(1) not in locus_tags_found:
                locus_tags_found.append(l_match.group(1))

    line_count = 0
    t0 = time.time()
    with open(path, "r", encoding=encoding, errors="replace") as fp:
        for line in fp:
            line_count += 1
            if line.startswith(">"):
                if current_header is not None:
                    seq_str = "".join(current_seq_chunks).upper()
                    process_record(current_header, seq_str)
                    current_seq_chunks = []
                current_header = line.strip()
            else:
                s = line.strip()
                if s:
                    current_seq_chunks.append(s)

            if line_count % progress_interval == 0:
                logger.info(
                    "Streaming %s: %d lines processed (%d records parsed)...",
                    path.name, line_count, total_records
                )

    if current_header is not None:
        seq_str = "".join(current_seq_chunks).upper()
        process_record(current_header, seq_str)

    if total_records == 0:
        is_valid_fasta = False
        min_len = 0

    avg_len = total_nucs / max(1, total_records)
    elapsed = time.time() - t0
    logger.info("Finished streaming %s: %d records in %.2fs", path.name, total_records, elapsed)

    # Percentage composition
    all_bases = {**base_counts, **other_bases}
    pcts = {
        b: round((cnt / max(1, total_nucs)) * 100, 2)
        for b, cnt in all_bases.items()
    }

    summary = FastaFileSummary(
        filename=path.name,
        extension=path.suffix,
        size_bytes=file_size,
        size_human=human_size,
        encoding=encoding,
        is_valid_fasta=is_valid_fasta,
        first_5_headers=first_5_headers,
        first_5_previews=first_5_previews,
        total_records=total_records,
        total_nucleotides=total_nucs,
        min_length=int(min_len) if min_len != float("inf") else 0,
        max_length=int(max_len),
        avg_length=round(avg_len, 2),
        alphabet_distribution=all_bases,
        alphabet_percentages=pcts,
        internal_duplicates=internal_duplicates,
        unique_sequences=len(seq_hashes),
        accession_prefixes=accession_prefixes,
        sample_genes=genes_found,
        sample_proteins=proteins_found,
        sample_locus_tags=locus_tags_found,
    )

    return summary, seq_hashes


def inspect_collection(
    fasta_dir: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Inspect all FASTA files in a directory and analyze cross-file sequence overlaps.

    Args:
        fasta_dir: Directory containing FASTA sequence files.
        output_json_path: Optional path to save full inspection results JSON.

    Returns:
        Dictionary containing file summaries and cross-file sequence overlaps.
    """
    directory = Path(fasta_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"FASTA directory not found: {directory}")

    files = sorted([f for f in directory.iterdir() if f.is_file()])
    logger.info("Found %d files in %s for streaming inspection", len(files), directory)

    summaries: Dict[str, FastaFileSummary] = {}
    hash_sets: Dict[str, Set[bytes]] = {}

    for f in files:
        summary, hashes = inspect_fasta_file(f)
        summaries[f.name] = summary
        hash_sets[f.name] = hashes

    # Compute pairwise cross-file sequence overlaps
    fnames = list(summaries.keys())
    cross_overlap: Dict[str, int] = {}
    for i in range(len(fnames)):
        for j in range(i + 1, len(fnames)):
            f1 = fnames[i]
            f2 = fnames[j]
            overlap_count = len(hash_sets[f1].intersection(hash_sets[f2]))
            cross_overlap[f"{f1} <-> {f2}"] = overlap_count

    output_data = {
        "files": {name: s.to_dict() for name, s in summaries.items()},
        "cross_file_overlap": cross_overlap,
        "total_collection_records": sum(s.total_records for s in summaries.values()),
        "total_collection_nucleotides": sum(s.total_nucleotides for s in summaries.values()),
        "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fp:
            json.dump(output_data, fp, indent=2)
        logger.info("Saved collection inspection report to %s", out_p)

    return output_data

