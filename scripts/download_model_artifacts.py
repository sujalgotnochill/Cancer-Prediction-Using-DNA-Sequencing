#!/usr/bin/env python
"""Model artifact verification, download, and packaging utility.

Manages distribution and cryptographic integrity verification of the frozen
Phase 5 classification model artifacts:
- model.joblib (LinearSVCClassifier)
- vectorizer.joblib (FeatureBuilder TF-IDF vectorizer, k=4, 256 tokens)
- encoder.joblib (LabelEncoder for 4 cancer classes)
- model_config.json (Frozen hyperparameters and CV benchmarks)
- checksums.sha256 (SHA-256 cryptographic hashes)

Usage:
    python scripts/download_model_artifacts.py [--verify-only] [--download] [--force]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EXPECTED_CHECKSUMS: Dict[str, str] = {
    "model.joblib": "1422d3163b1d33bb84052f40bbb7466476a36ecadf85dd440d6278844faff386",
    "vectorizer.joblib": "a034de878fac4ba7401f13555b4b5aa6509e11fc282754ae526645b2668093c8",
    "encoder.joblib": "3c78f7eee9e9be78349cf8a3c3bcee02a0efaad961252f727bb2b43065c9015c",
    "model_config.json": "35fe8ec22804993160e649cd19c82e2270f4bec09840099f8efa7a8dc0984aae",
}

DEFAULT_MODEL_DIR = Path("artifacts/models/phase5_best_candidate")
DEFAULT_REMOTE_URL_ENV = "MODEL_ARTIFACTS_URL"


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hexadecimal hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def load_checksums_file(checksums_path: Path) -> Dict[str, str]:
    """Parse a standard sha256sum formatted checksums file."""
    checksums: Dict[str, str] = {}
    if not checksums_path.exists():
        return checksums
    with open(checksums_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                sha, fname = parts[0].strip().lower(), Path(parts[1].strip()).name
                checksums[fname] = sha
    return checksums


def write_checksums_file(model_dir: Path, checksums: Dict[str, str]) -> Path:
    """Generate or update the checksums.sha256 file in the model directory."""
    out_path = model_dir / "checksums.sha256"
    lines = [f"{sha}  {fname}\n" for fname, sha in sorted(checksums.items())]
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_path


def verify_artifacts(
    model_dir: Path,
    expected_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[bool, List[str]]:
    """Verify that all required artifact files exist and match their expected SHA-256 hashes.

    Returns:
        (is_valid, list_of_error_messages)
    """
    targets = expected_hashes or EXPECTED_CHECKSUMS
    errors: List[str] = []

    if not model_dir.exists():
        return False, [f"Model directory does not exist: {model_dir}"]

    checksums_file = model_dir / "checksums.sha256"
    file_hashes = load_checksums_file(checksums_file) if checksums_file.exists() else {}

    for fname, expected_sha in targets.items():
        fpath = model_dir / fname
        if not fpath.exists():
            errors.append(f"Missing artifact: {fname} (expected at {fpath})")
            continue

        actual_sha = compute_sha256(fpath)
        # Compare against hardcoded canonical hash first, then checksums.sha256 if present
        ref_sha = expected_sha.lower()
        if actual_sha != ref_sha:
            errors.append(
                f"Checksum mismatch for {fname}:\n"
                f"  Expected: {ref_sha}\n"
                f"  Actual:   {actual_sha}"
            )
        elif fname in file_hashes and file_hashes[fname].lower() != actual_sha:
            errors.append(
                f"checksums.sha256 entry mismatch for {fname}:\n"
                f"  File record: {file_hashes[fname]}\n"
                f"  Actual hash: {actual_sha}"
            )

    return len(errors) == 0, errors


def download_artifacts_archive(
    url: str,
    target_dir: Path,
    expected_hashes: Optional[Dict[str, str]] = None,
) -> bool:
    """Download a zip/tar archive containing model artifacts and extract into target_dir."""
    print(f"Downloading model artifacts from: {url}")
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = target_dir / "temp_artifacts.zip"

    try:
        urllib.request.urlretrieve(url, str(temp_zip))
        print("Download complete. Extracting archive...")
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(target_dir)
        temp_zip.unlink(missing_ok=True)

        is_valid, errors = verify_artifacts(target_dir, expected_hashes)
        if not is_valid:
            print("Extracted artifacts failed integrity verification:")
            for err in errors:
                print(f"  - {err}")
            return False

        print("Artifacts successfully downloaded and verified!")
        return True
    except Exception as exc:
        print(f"Failed to download or extract artifacts: {exc}", file=sys.stderr)
        if temp_zip.exists():
            temp_zip.unlink(missing_ok=True)
        return False


def package_artifacts_zip(model_dir: Path, output_zip: Path) -> Path:
    """Package the verified model artifacts into a distributable zip file."""
    is_valid, errors = verify_artifacts(model_dir)
    if not is_valid:
        raise ValueError(f"Cannot package invalid artifacts: {errors}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in list(EXPECTED_CHECKSUMS.keys()) + ["checksums.sha256"]:
            fpath = model_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
    print(f"Packaged verified artifacts into: {output_zip} ({output_zip.stat().st_size} bytes)")
    return output_zip


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or set up frozen Phase 5 model artifacts.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory holding model artifacts (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing artifacts without attempting download.",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=os.environ.get(DEFAULT_REMOTE_URL_ENV),
        help=f"Custom URL to download model artifacts archive from (or set ${DEFAULT_REMOTE_URL_ENV}).",
    )
    parser.add_argument(
        "--package-zip",
        type=Path,
        default=None,
        help="Package existing verified artifacts into a distributable zip file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (exit code 0 indicates valid).",
    )

    args = parser.parse_args()
    model_dir: Path = args.model_dir

    if args.package_zip:
        try:
            package_artifacts_zip(model_dir, args.package_zip)
            return 0
        except Exception as exc:
            print(f"Error packaging artifacts: {exc}", file=sys.stderr)
            return 1

    # Ensure checksums.sha256 is present if all artifacts exist
    all_files_exist = all((model_dir / f).exists() for f in EXPECTED_CHECKSUMS)
    if all_files_exist and not (model_dir / "checksums.sha256").exists():
        write_checksums_file(model_dir, EXPECTED_CHECKSUMS)

    is_valid, errors = verify_artifacts(model_dir)

    if is_valid:
        if not args.quiet:
            print(f"All Phase 5 model artifacts in '{model_dir}' are verified and intact:")
            for fname, sha in EXPECTED_CHECKSUMS.items():
                size = (model_dir / fname).stat().st_size
                print(f"  [OK] {fname:<20} ({size:>6} bytes) SHA-256: {sha[:16]}...")
        return 0

    if not args.quiet:
        print("Phase 5 model artifact verification status: INCOMPLETE or CORRUPTED")
        for err in errors:
            print(f"  ! {err}")

    if args.verify_only:
        return 1

    # Attempt download if URL is provided
    if args.download_url:
        success = download_artifacts_archive(args.download_url, model_dir)
        return 0 if success else 1

    if not args.quiet:
        print("\nSetup Instructions:")
        print(f"  The model artifacts (~20 KB total) should reside in: {model_dir}")
        print("  Expected files:")
        for fname, sha in EXPECTED_CHECKSUMS.items():
            print(f"    - {fname:<20} (SHA-256: {sha})")
        print("\n  If you obtained a release archive, extract it into the model directory:")
        print(f"    python scripts/download_model_artifacts.py --download-url <URL_TO_RELEASE_ZIP>")
        print("\n  Or set the environment variable:")
        print(f"    $env:{DEFAULT_REMOTE_URL_ENV} = 'https://.../artifacts.zip'")
        print("    python scripts/download_model_artifacts.py")

    return 1


if __name__ == "__main__":
    sys.exit(main())

