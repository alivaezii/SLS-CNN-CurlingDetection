# 01_split_auto_safe_raw_v5.py
# From-raw temporal split with:
# - De-duplication by filename STEM (case-insensitive; prefers .jpg > .jpeg > .png > .bmp)
# - Robust frame index: take the *last* numeric group in the filename stem
# - Anti-leakage: enforce SAME-class temporal gaps (>= min_gap_same)
# - Tuned gaps: gap_tv=0 to recover Gap#1, gap_vt=-50 to shrink Gap#2 while keeping safety

from pathlib import Path
import re
import shutil
import numpy as np

# -------------------- CONFIG --------------------
ROOT = Path(r"G:\Frames")  # base folder containing raw class folders

# Raw folders (positives/negatives)
RAW_POS = ROOT / "Frames mit Curling 06.04.2022"
RAW_NEG = ROOT / "Frames ohne Curling 06.04.2022"

# Output (fresh folder; do NOT reuse older outputs)
OUT = ROOT / "dataset_final_RawV5"

# Target class subfolders
TRAIN_POS = OUT / "train" / "curling"
VAL_POS   = OUT / "val"   / "curling"
TEST_POS  = OUT / "test"  / "curling"
TRAIN_NEG = OUT / "train" / "ohne_curling"
VAL_NEG   = OUT / "val"   / "ohne_curling"
TEST_NEG  = OUT / "test"  / "ohne_curling"

# Allowed extensions (case-insensitive) and preference order for de-dup
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PREF_ORDER   = {".jpg": 0, ".jpeg": 1, ".png": 2, ".bmp": 3}

# ---------------- Helpers ----------------
def frame_index(p: Path) -> int:
    """
    Extract the *last* numeric group from the filename stem.
    Examples:
      'Video_...T121928676.avi0.jpg'   -> 0
      'Video_...T121928676.avi111.jpg' -> 111
      'img_000123.png'                 -> 123
    Returns -1 if no digits exist.
    """
    s = p.stem
    m_end = re.search(r"(\d+)$", s)         # digits at the very end
    if m_end:
        return int(m_end.group(1))
    m_all = re.findall(r"(\d+)", s)         # otherwise last numeric group anywhere
    return int(m_all[-1]) if m_all else -1

def rglob_images(folder: Path):
    """Recursively list images with allowed extensions (case-insensitive)."""
    out = []
    if not folder.exists():
        return out
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
            out.append(f)
    return out

def list_images_dedup_by_stem(folder: Path):
    """
    Collect images and de-duplicate by filename stem (case-insensitive).
    Keep exactly one file per stem; preference: .jpg > .jpeg > .png > .bmp.
    """
    pool = rglob_images(folder)
    best = {}   # stem_lower -> (priority, path)
    dropped = 0
    for p in pool:
        stem = p.stem.lower()
        pr   = PREF_ORDER.get(p.suffix.lower(), 99)
        cur  = best.get(stem)
        if (cur is None) or (pr < cur[0]):
            best[stem] = (pr, p)
        else:
            dropped += 1
    files = [p for _, p in best.values()]
    return files, len(pool), dropped

def empty_dir_keep(d: Path):
    """Clear files under d (if any) and ensure the directory exists."""
    if d.exists():
        for x in d.rglob("*"):
            try:
                if x.is_file() or x.is_symlink():
                    x.unlink()
            except:
                pass
    d.mkdir(parents=True, exist_ok=True)

def safe_copy(files, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst / src.name)

# ---------------- Split helpers ----------------
def ranges_from_curling(curl_idxs, all_idxs, q_low, q_high, gap_tv, gap_vt):
    """Build train/val/test windows driven by curling quantiles."""
    min_frame, max_frame = int(all_idxs[0]), int(all_idxs[-1])
    qL = int(np.floor(np.quantile(curl_idxs, q_low)))
    qH = int(np.floor(np.quantile(curl_idxs, q_high)))

    # train: [min, qL]
    train = (min_frame, qL)

    # val: [qL + gap_tv, qH - gap_tv] (relax around mid if too tight)
    val_lo = qL + max(0, gap_tv)
    val_hi = qH - max(1, gap_tv)
    if val_lo > val_hi:
        mid = (qL + qH) // 2
        val_lo, val_hi = min(mid, qH - 1), max(mid + 1, qH)
    val = (val_lo, val_hi)

    # test: [qH + gap_vt, max]
    test_lo = qH + gap_vt
    if test_lo > max_frame:
        test_lo = qH
    test = (test_lo, max_frame)

    return train, val, test, (qL, qH)

def assign_by_ranges(pos, neg, train_rng, val_rng, test_rng):
    def in_rng(idx, rng): return rng[0] <= idx <= rng[1]
    tr_p, va_p, te_p, tr_n, va_n, te_n = [], [], [], [], [], []
    for p in pos:
        idx = frame_index(p)
        if idx < 0: continue
        if   in_rng(idx, train_rng): tr_p.append(p)
        elif in_rng(idx, val_rng):   va_p.append(p)
        elif in_rng(idx, test_rng):  te_p.append(p)
    for n in neg:
        idx = frame_index(n)
        if idx < 0: continue
        if   in_rng(idx, train_rng): tr_n.append(n)
        elif in_rng(idx, val_rng):   va_n.append(n)
        elif in_rng(idx, test_rng):  te_n.append(n)
    return tr_p, va_p, te_p, tr_n, va_n, te_n

# ---------------- Min-gap utilities ----------------
def min_gap_two_pointer(sorted_a, sorted_b):
    """Compute min |a-b| for two sorted lists in O(n+m)."""
    if not sorted_a or not sorted_b:
        return np.inf
    i = j = 0
    best = np.inf
    while i < len(sorted_a) and j < len(sorted_b):
        da, db = sorted_a[i], sorted_b[j]
        diff = abs(da - db)
        if diff < best:
            best = diff
            if best == 0:
                return 0
        if da < db:
            i += 1
        else:
            j += 1
    return int(best)

def build_index_sets(tr_p, va_p, te_p, tr_n, va_n, te_n):
    return {
        "train_curling":  sorted({frame_index(p) for p in tr_p if frame_index(p) >= 0}),
        "val_curling":    sorted({frame_index(p) for p in va_p if frame_index(p) >= 0}),
        "test_curling":   sorted({frame_index(p) for p in te_p if frame_index(p) >= 0}),
        "train_ohne":     sorted({frame_index(p) for p in tr_n if frame_index(p) >= 0}),
        "val_ohne":       sorted({frame_index(p) for p in va_n if frame_index(p) >= 0}),
        "test_ohne":      sorted({frame_index(p) for p in te_n if frame_index(p) >= 0}),
    }

def report_gaps(sets, min_gap_same=200, min_gap_cross_warn=100):
    """Enforce SAME-class gaps; only warn for CROSS-class."""
    ok = True
    lines = []

    # SAME-class pairs (mandatory)
    enforce_pairs = [
        ("train_curling","val_curling"),
        ("train_curling","test_curling"),
        ("val_curling","test_curling"),
        ("train_ohne","val_ohne"),
        ("train_ohne","test_ohne"),
        ("val_ohne","test_ohne"),
    ]
    for a, b in enforce_pairs:
        d = min_gap_two_pointer(sets[a], sets[b])
        if d < min_gap_same:
            ok = False
            lines.append(f"❌ SAME {a} ↔ {b}: min Δframe={d} (<{min_gap_same})")
        else:
            lines.append(f"✅ SAME {a} ↔ {b}: min Δframe={d}")

    # CROSS-class (advisory)
    warn_pairs = [
        ("train_curling","val_ohne"),
        ("train_curling","test_ohne"),
        ("val_curling","train_ohne"),
        ("val_curling","test_ohne"),
        ("test_curling","train_ohne"),
        ("test_curling","val_ohne"),
    ]
    for a, b in warn_pairs:
        d = min_gap_two_pointer(sets[a], sets[b])
        if d < min_gap_cross_warn:
            lines.append(f"⚠️ CROSS {a} ↔ {b}: min Δframe={d} (<{min_gap_cross_warn})")
        else:
            lines.append(f"✅ CROSS {a} ↔ {b}: min Δframe={d}")

    return ok, "\n".join(lines)

# ---------------- Orchestrator ----------------
def main(
    min_pos_val=30, min_pos_test=30,
    q_low=0.44, q_high=0.76,
    gap_tv=0,         # close Gap#1
    gap_vt=-50,       # shrink Gap#2 (script still enforces min_gap_same)
    min_gap_same=200, min_gap_cross_warn=100,
    max_attempts=30
):
    # Collect & de-dup by STEM
    pos, pos_raw_cnt, pos_dropped = list_images_dedup_by_stem(RAW_POS)
    neg, neg_raw_cnt, neg_dropped = list_images_dedup_by_stem(RAW_NEG)

    print(f"Raw images (found) → curling: {pos_raw_cnt} | ohne: {neg_raw_cnt}")
    print(f"After de-dup by filename stem → curling: {len(pos)} (dropped {pos_dropped}) | ohne: {len(neg)} (dropped {neg_dropped})")

    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Raw input folders empty or wrong paths/extensions.")

    # Keep only items with a valid frame index
    valid_pos = [p for p in pos if frame_index(p) >= 0]
    valid_neg = [n for n in neg if frame_index(n) >= 0]
    if len(valid_pos) != len(pos) or len(valid_neg) != len(neg):
        print(f"Note: skipped (no frame index) → curling: {len(pos)-len(valid_pos)}, ohne: {len(neg)-len(valid_neg)}")
    pos, neg = valid_pos, valid_neg

    curl_idxs = sorted({frame_index(p) for p in pos})
    all_idxs  = sorted({frame_index(p) for p in pos + neg})
    if len(curl_idxs) < 10:
        raise RuntimeError("Too few curling indices parsed (<10). Check filenames.")

    # Prepare fresh output dirs
    for d in [TRAIN_POS, VAL_POS, TEST_POS, TRAIN_NEG, VAL_NEG, TEST_NEG]:
        empty_dir_keep(d)

    attempts, tried = 0, set()
    while attempts < max_attempts:
        attempts += 1
        key = (round(q_low,3), round(q_high,3), gap_tv, gap_vt)
        if key in tried:
            # small nudges to escape loops
            q_low  = max(0.30, q_low - 0.01)
            q_high = min(0.85, q_high + 0.01)
        tried.add(key)

        train_rng, val_rng, test_rng, (qL, qH) = ranges_from_curling(
            curl_idxs, all_idxs, q_low, q_high, gap_tv, gap_vt
        )

        tr_p, va_p, te_p, tr_n, va_n, te_n = assign_by_ranges(pos, neg, train_rng, val_rng, test_rng)

        print(f"\nAttempt {attempts} | q=({q_low:.2f},{q_high:.2f}) gaps(tv={gap_tv}, vt={gap_vt})")
        print(f"Ranges: train={train_rng}  val={val_rng}  test={test_rng}")
        print(f"Counts: curling(train={len(tr_p)}, val={len(va_p)}, test={len(te_p)}), "
              f"ohne(train={len(tr_n)}, val={len(va_n)}, test={len(te_n)})")

        # Ensure enough positives in val/test
        if len(va_p) < min_pos_val or len(te_p) < min_pos_test:
            print("→ Adjusting to ensure positive samples in val/test ...")
            if len(te_p) < min_pos_test:
                if gap_vt > -10:
                    gap_vt = max(-10, gap_vt - 1)   # pull test earlier
                else:
                    q_high = max(0.66, q_high - 0.02)
                continue
            if len(va_p) < min_pos_val:
                if gap_tv > 0:
                    gap_tv = max(0, gap_tv // 2)
                else:
                    q_low  = max(0.30, q_low - 0.02)
                    q_high = min(0.85, q_high + 0.01)
                continue

        # Enforce SAME-class temporal safety
        sets = build_index_sets(tr_p, va_p, te_p, tr_n, va_n, te_n)
        ok_same, gap_report = report_gaps(sets, min_gap_same=min_gap_same, min_gap_cross_warn=min_gap_cross_warn)
        print(gap_report)
        if not ok_same:
            print("→ Adjusting to increase SAME-class temporal safety ...")
            gap_tv = min(800, gap_tv + 50)
            gap_vt = min(800, max(gap_vt, 0) + 25)
            q_low  = min(0.48, q_low + 0.01)
            q_high = max(0.70, q_high - 0.01)
            continue

        # All good → write files
        print("\n✅ Split passed counts & SAME-class leakage checks. Writing files...")
        for d in [TRAIN_POS, VAL_POS, TEST_POS, TRAIN_NEG, VAL_NEG, TEST_NEG]:
            empty_dir_keep(d)
        safe_copy(tr_p, TRAIN_POS); safe_copy(va_p, VAL_POS); safe_copy(te_p, TEST_POS)
        safe_copy(tr_n, TRAIN_NEG); safe_copy(va_n, VAL_NEG); safe_copy(te_n, TEST_NEG)

        print("\nFINAL SUMMARY")
        print(f"Quantiles on curling: q_low_idx≈{qL}, q_high_idx≈{qH}")
        print(f"Final params: q=({q_low:.2f},{q_high:.2f}) gaps(tv={gap_tv}, vt={gap_vt}) "
              f"min_gap_same={min_gap_same} min_gap_cross_warn={min_gap_cross_warn}")
        print(f"Final ranges: train={train_rng}  val={val_rng}  test={test_rng}")
        print(f"Final counts: curling(train={len(tr_p)}, val={len(va_p)}, test={len(te_p)}), "
              f"ohne(train={len(tr_n)}, val={len(va_n)}, test={len(te_n)})")
        print("✅ All good. You can proceed to audit.")
        return

    raise RuntimeError("Failed to satisfy counts/leakage within max_attempts. Consider relaxing min_gap_same or targets.")

if __name__ == "__main__":
    main()
