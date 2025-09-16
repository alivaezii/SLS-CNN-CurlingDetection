# leakagecheck02.py  — self-contained (auto-installs numpy + pillow if missing)
# Purpose: Build per-split manifests (train/val/test) and detect cross-split duplicates / near-duplicates.
# Output: G:\procnn\leakage_reports\{manifest.tsv, overlap_*.tsv, summary.json}

import sys, subprocess, importlib, os, csv, hashlib, json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------- Auto-install minimal deps (no venv needed) -----------------
def _pip_install(pkg: str) -> None:
    """
    Try to install a package using pip. If the first attempt fails,
    retries with '--user' (useful on Windows without admin).
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

def _install_and_import(pkg: str, import_name: str = None):
    """
    Import a module; if unavailable, install via pip then import again.
    import_name lets you pip-install 'pillow' but import 'PIL'.
    """
    name = import_name or pkg
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"[INFO] '{name}' not found. Installing '{pkg}' ...")
        _pip_install(pkg)
        return importlib.import_module(name)

np_module = _install_and_import("numpy", "numpy")
PIL_module = _install_and_import("pillow", "PIL")
from PIL import Image, ImageFile  # safe now
import numpy as np                # alias for convenience

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------- Config -----------------
PROJ_ROOT = Path(r"G:\procnn")  # adjust if needed
DATA_ROOT = PROJ_ROOT / "dataset_final_RawV5"  # expected: train/ val/ test/ with class subfolders
OUT_DIR   = PROJ_ROOT / "leakage_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
LABELS = ["ohne_curling", "curling"]   # adapt if more classes exist
MANIFEST_NAME = "manifest.tsv"
AHASH_SIZE = 8      # 8x8 -> 64-bit aHash
NEAR_DUP_HAMMING_MAX = 4  # <=4 bits difference is considered near-duplicate

# ----------------- Utils -----------------
def is_image(p: Path) -> bool:
    exts = {
        ".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif",
        ".JPG",".JPEG",".PNG",".BMP",".WEBP",".TIF",".TIFF",".GIF"
    }
    return p.is_file() and p.suffix in exts

def sha256_of_file(p: Path, chunk=1024*1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ahash_of_image(p: Path, size: int = AHASH_SIZE) -> str:
    """
    Average hash (aHash): resize -> grayscale -> threshold by mean.
    Returns 64-bit hex string (for 8x8), upper-case. Returns "" on failure.
    """
    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((size, size))
            a = np.asarray(im, dtype=np.float32)
            m = a.mean()
            bits = (a >= m).astype(np.uint8).flatten()
            val = 0
            for b in bits:
                val = (val << 1) | int(b)
            return f"{val:0{size*size//4}X}"
    except Exception:
        return ""  # unreadable -> empty

def hamming_hex(h1: str, h2: str) -> int:
    if not h1 or not h2 or len(h1) != len(h2):
        return 999
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

def write_manifest(rows: List[dict], out_path: Path):
    cols = ["split","label","relpath","filename","size","sha256","ahash"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

# ----------------- Build manifests -----------------
if not DATA_ROOT.exists():
    raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

all_rows: List[dict] = []
split_counts: Dict[str, int] = {}

for split in SPLITS:
    base = DATA_ROOT / split
    if not base.exists():
        raise FileNotFoundError(f"Split not found: {base}")
    for label in LABELS:
        d = base / label
        if not d.exists():
            # tolerate missing class folder but log it
            print(f"[WARN] Missing class folder: {d}")
            continue
        for p in d.rglob("*"):
            if not is_image(p):
                continue
            rel = p.relative_to(DATA_ROOT).as_posix()
            size = p.stat().st_size
            sha  = sha256_of_file(p)
            ah   = ahash_of_image(p)
            all_rows.append({
                "split": split,
                "label": label,
                "relpath": rel,
                "filename": p.name,
                "size": str(size),
                "sha256": sha,
                "ahash": ah
            })

manifest_path = OUT_DIR / MANIFEST_NAME
write_manifest(all_rows, manifest_path)

for split in SPLITS:
    rows = [r for r in all_rows if r["split"] == split]
    write_manifest(rows, OUT_DIR / f"{split}_{MANIFEST_NAME}")
    split_counts[split] = len(rows)

print("[OK] Manifests written:")
for k, v in split_counts.items():
    print(f"  - {k}: {v} files")
print(f"  -> {manifest_path}")

# ----------------- Cross-split overlap checks -----------------
# 1) Exact duplicates by sha256 across splits
rows_by_sha: Dict[str, List[dict]] = defaultdict(list)
for r in all_rows:
    rows_by_sha[r["sha256"]].append(r)

exact_dupes = []
for sha, rows in rows_by_sha.items():
    splits_present = set([r["split"] for r in rows])
    if len(splits_present) >= 2:
        rows_sorted = sorted(rows, key=lambda x: (x["split"], x["relpath"]))
        for i in range(len(rows_sorted)):
            for j in range(i + 1, len(rows_sorted)):
                if rows_sorted[i]["split"] != rows_sorted[j]["split"]:
                    exact_dupes.append((rows_sorted[i], rows_sorted[j]))

# 2) Weak duplicates by (filename, size) across splits
key2rows: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
for r in all_rows:
    key = (r["filename"], int(r["size"]))
    key2rows[key].append(r)

name_size_hits = []
for key, rows in key2rows.items():
    splits_present = set([r["split"] for r in rows])
    if len(splits_present) >= 2:
        rows_sorted = sorted(rows, key=lambda x: (x["split"], x["relpath"]))
        for i in range(len(rows_sorted)):
            for j in range(i + 1, len(rows_sorted)):
                if rows_sorted[i]["split"] != rows_sorted[j]["split"]:
                    name_size_hits.append((rows_sorted[i], rows_sorted[j]))

# 3) Near-duplicates by aHash (Hamming distance <= NEAR_DUP_HAMMING_MAX)
split_rows = {s: [r for r in all_rows if r["split"] == s and r["ahash"]] for s in SPLITS}
near_dupes = []
pairs_checked = 0
for a in SPLITS:
    for b in SPLITS:
        if a >= b:  # ensure ordered pairs: train<val<test
            continue
        A, B = split_rows[a], split_rows[b]
        for ra in A:
            ah_a = ra["ahash"]
            for rb in B:
                pairs_checked += 1
                d = hamming_hex(ah_a, rb["ahash"])
                if d <= NEAR_DUP_HAMMING_MAX:
                    near_dupes.append((d, ra, rb))

print(f"[INFO] near-duplicate comparisons: {pairs_checked}")

# ----------------- Write reports -----------------
def write_pairs(pairs, out_path: Path, title: str):
    cols = ["split_a","label_a","relpath_a","size_a","sha256_a","ahash_a",
            "split_b","label_b","relpath_b","size_b","sha256_b","ahash_b","extra"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["# " + title])
        w.writerow(cols)
        for item in pairs:
            if isinstance(item[0], dict):
                a, b = item
                extra = ""
            else:
                dist, a, b = item
                extra = f"hamming={dist}"
            w.writerow([a["split"], a["label"], a["relpath"], a["size"], a["sha256"], a["ahash"],
                        b["split"], b["label"], b["relpath"], b["size"], b["sha256"], b["ahash"],
                        extra])

exact_path     = OUT_DIR / "overlap_exact_sha256.tsv"
name_size_path = OUT_DIR / "overlap_name_size.tsv"
near_path      = OUT_DIR / f"near_dupes_ahash_le_{NEAR_DUP_HAMMING_MAX}.tsv"

write_pairs([(a, b) for a, b in exact_dupes], exact_path, "Exact duplicates by SHA256 across splits")
write_pairs([(a, b) for a, b in name_size_hits], name_size_path, "Same filename+size across splits (possible duplicates)")
write_pairs(near_dupes, near_path, f"Near-duplicates by aHash (Hamming ≤ {NEAR_DUP_HAMMING_MAX})")

summary = {
    "counts": split_counts,
    "reports": {
        "manifest_all": str(manifest_path),
        "overlap_exact_sha256": str(exact_path),
        "overlap_name_size": str(name_size_path),
        "near_dupes_ahash": str(near_path)
    },
    "params": {
        "ahash_size": AHASH_SIZE,
        "near_dupe_hamming_max": NEAR_DUP_HAMMING_MAX
    }
}
with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== Leakage Check Summary ===")
print(json.dumps(summary, indent=2))
print("\nTips:")
print("- If any of the TSV reports are non-empty, there is potential cross-split leakage.")
print("- Prefer fixing by re-generating the split to enforce zero overlaps; then re-train.")
