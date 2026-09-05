"""
build_merged_dataset.py
=======================
Build merged Hateful Memes (Facebook AI) + Memotion 7k dataset.

Why merge?
-----------
- FB dataset: 10000 high-quality samples (limited diversity).
- Memotion 7k: 6992 noisier samples (adds humour/sarcasm/offensive variety).
- Combined: stronger generalization across meme styles.

Why uniform 25% Memotion in all splits?
---------------------------------------
Empirical baseline (Qwen2VL frozen):
  - FB only:                                        AUROC 0.8528 (dev)
  - Earlier merge (imbalanced 13.7% / 37.5% / 16.7% Memotion):
                                                   AUROC 0.6532 (dev)

Root cause: distribution mismatch between train and dev/test Memotion ratio.
Fix:        uniform 25% Memotion across train/dev/test so the model sees the
            same mixture at training and evaluation time.

Why 25% specifically?
- Domain-adaptation literature: 20-30 % target-domain data is the sweet spot
  to learn a second distribution without primary-domain drift.
- Dev set keeps 667 samples (>= 600) for stable AUROC estimation.
- Class balance after merge stays close to FB's natural pos% range.
- Uses ~48% of available Memotion data (3333 / 6992).

Output layout (FB-compatible)
-----------------------------
merge_dataset/data/
├── train.jsonl  # {id: int, img: "img/<id>.<ext>", label: 0/1, text: str}
├── dev.jsonl
├── test.jsonl
└── img/         # flat folder: FB (5-digit) + Memotion (7-digit) images

ID convention
-------------
- FB ids:       1..98764 (5-digit zero-padded filename in img/)
- Memotion ids: 1000000 + N where N is parsed from 'image_N.jpg|png|...'
  No collision with FB ids since FB max < 100 000 < 1 000 000.

Reproducibility
---------------
random.seed(42). Stratified-by-label sampling preserves Memotion's 24% pos
rate inside each split.
"""

import json
import random
import re
import shutil
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
FB_DATA_DIR = PROJECT_ROOT / "data"
MEMOTION_DIR = PROJECT_ROOT / "memotion_dataset_7k"
OUT_DATA_DIR = PROJECT_ROOT / "merge_dataset" / "data"
OUT_IMG_DIR = OUT_DATA_DIR / "img"

TARGET_MEMOTION_RATIO = 0.25       # 25% Memotion in every split
MEMOTION_ID_OFFSET = 1_000_000     # new_int_id = OFFSET + memotion_num
SEED = 42


# ============================================================
# Helpers
# ============================================================
def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def memotion_num_from_id(memotion_id: str) -> int:
    m = re.search(r"image_(\d+)", memotion_id)
    if not m:
        raise ValueError(f"Cannot parse Memotion id: {memotion_id!r}")
    return int(m.group(1))


def target_memotion_count(fb_n: int, ratio: float) -> int:
    """How many Memotion samples to add to a split of FB size fb_n so that
    memo / (fb_n + memo) == ratio."""
    return round(fb_n * ratio / (1.0 - ratio))


# ============================================================
# Step 1 — load source data
# ============================================================
print("=" * 60)
print("STEP 1: Load source datasets")
print("=" * 60)
fb_train = load_jsonl(FB_DATA_DIR / "train.jsonl")
fb_dev = load_jsonl(FB_DATA_DIR / "dev.jsonl")
fb_test = load_jsonl(FB_DATA_DIR / "test.jsonl")
print(f"FB:       train={len(fb_train):5d}, dev={len(fb_dev):4d}, test={len(fb_test):4d}")

mem_all = (
    load_jsonl(MEMOTION_DIR / "data" / "train.jsonl")
    + load_jsonl(MEMOTION_DIR / "data" / "dev.jsonl")
    + load_jsonl(MEMOTION_DIR / "data" / "test.jsonl")
)
mem_pos = sum(1 for r in mem_all if r["label"] == 1)
print(f"Memotion: pool ={len(mem_all):5d} (pos={mem_pos}, {mem_pos*100/len(mem_all):.1f}%)")


# ============================================================
# Step 2 — compute target Memotion counts (25% uniform)
# ============================================================
print()
print("=" * 60)
print("STEP 2: Compute target Memotion counts (uniform 25%)")
print("=" * 60)
n_train_mem = target_memotion_count(len(fb_train), TARGET_MEMOTION_RATIO)
n_dev_mem = target_memotion_count(len(fb_dev), TARGET_MEMOTION_RATIO)
n_test_mem = target_memotion_count(len(fb_test), TARGET_MEMOTION_RATIO)
print(f"  train: +{n_train_mem} Memotion -> total {len(fb_train)+n_train_mem} "
      f"({n_train_mem*100/(len(fb_train)+n_train_mem):.1f}% Memotion)")
print(f"  dev:   +{n_dev_mem} Memotion -> total {len(fb_dev)+n_dev_mem} "
      f"({n_dev_mem*100/(len(fb_dev)+n_dev_mem):.1f}% Memotion)")
print(f"  test:  +{n_test_mem} Memotion -> total {len(fb_test)+n_test_mem} "
      f"({n_test_mem*100/(len(fb_test)+n_test_mem):.1f}% Memotion)")
total_needed = n_train_mem + n_dev_mem + n_test_mem
print(f"  Memotion usage: {total_needed}/{len(mem_all)} = {total_needed*100/len(mem_all):.1f}%")
assert total_needed <= len(mem_all), "Memotion pool too small"


# ============================================================
# Step 3 — stratified sample Memotion (preserve 24% pos rate)
# ============================================================
print()
print("=" * 60)
print("STEP 3: Stratified sample Memotion (seed=42, label-balanced)")
print("=" * 60)
random.seed(SEED)
random.shuffle(mem_all)
pos_pool = [r for r in mem_all if r["label"] == 1]
neg_pool = [r for r in mem_all if r["label"] == 0]
print(f"Pool after shuffle: pos={len(pos_pool)}, neg={len(neg_pool)}")


def stratified_pop(n: int) -> list:
    """Pop n samples from (pos_pool, neg_pool) preserving current pos:neg ratio."""
    remaining = len(pos_pool) + len(neg_pool)
    n_pos = round(n * len(pos_pool) / remaining)
    n_neg = n - n_pos
    n_pos = min(n_pos, len(pos_pool))
    n_neg = min(n_neg, len(neg_pool))
    out = pos_pool[:n_pos] + neg_pool[:n_neg]
    del pos_pool[:n_pos]
    del neg_pool[:n_neg]
    random.shuffle(out)
    return out


mem_split = {
    "train": stratified_pop(n_train_mem),
    "dev": stratified_pop(n_dev_mem),
    "test": stratified_pop(n_test_mem),
}
for split, lst in mem_split.items():
    pos = sum(1 for r in lst if r["label"] == 1)
    print(f"  {split}: {len(lst)} samples (pos={pos}, {pos*100/len(lst):.1f}%)")


# ============================================================
# Step 4 — convert Memotion records to FB int-ID format
# ============================================================
def memotion_to_fb(rec: dict):
    """Return (new_record, src_filename, dst_filename)."""
    num = memotion_num_from_id(rec["id"])
    ext = Path(rec["id"]).suffix
    new_id = MEMOTION_ID_OFFSET + num
    return (
        {
            "id": new_id,
            "img": f"img/{new_id}{ext}",
            "label": int(rec["label"]),
            "text": rec.get("text", ""),
        },
        f"image_{num}{ext}",
        f"{new_id}{ext}",
    )


def normalize_fb(rec: dict) -> dict:
    """Normalize FB record. FB test.jsonl uses 'test/<id>.png' prefix instead of
    'img/<id>.png' (inconsistency in the original release), so rewrite it to the
    flat 'img/' layout we use here."""
    img = rec["img"]
    fname = Path(img).name
    return {
        "id": int(rec["id"]),
        "img": f"img/{fname}",
        "label": int(rec["label"]),
        "text": rec.get("text", ""),
    }


# ============================================================
# Step 5 — build merged splits
# ============================================================
print()
print("=" * 60)
print("STEP 4: Build merged splits")
print("=" * 60)
merged = {
    "train": [normalize_fb(r) for r in fb_train],
    "dev": [normalize_fb(r) for r in fb_dev],
    "test": [normalize_fb(r) for r in fb_test],
}
images_to_copy = []  # (src_path, dst_filename)

for split, mlist in mem_split.items():
    for rec in mlist:
        new_rec, src_fn, dst_fn = memotion_to_fb(rec)
        merged[split].append(new_rec)
        images_to_copy.append((MEMOTION_DIR / "images" / src_fn, dst_fn))

for split in merged:
    random.shuffle(merged[split])
    pos = sum(1 for r in merged[split] if r["label"] == 1)
    mem = sum(1 for r in merged[split] if r["id"] >= MEMOTION_ID_OFFSET)
    n = len(merged[split])
    print(f"  {split}: total={n:5d}, pos={pos} ({pos*100/n:.1f}%), "
          f"Memotion={mem} ({mem*100/n:.1f}%)")


# ============================================================
# Step 6 — write JSONL (overwrite existing)
# ============================================================
print()
print("=" * 60)
print("STEP 5: Write JSONL files (overwrite)")
print("=" * 60)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
for split, records in merged.items():
    out_path = OUT_DATA_DIR / f"{split}.jsonl"
    save_jsonl(out_path, records)
    print(f"  wrote {out_path} ({len(records)} records)")


# ============================================================
# Step 7 — cleanup legacy Memotion-named files in img/
# ============================================================
print()
print("=" * 60)
print("STEP 6: Cleanup legacy 'image_X.<ext>' files in img/")
print("=" * 60)
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
legacy_pat = re.compile(r"^image_\d+\.(jpg|jpeg|png|bmp|jpe)$", re.IGNORECASE)
removed = 0
for f in OUT_IMG_DIR.iterdir():
    if legacy_pat.match(f.name):
        f.unlink()
        removed += 1
print(f"  removed {removed} legacy files")


# ============================================================
# Step 8 — copy Memotion images with new int-ID filename
# ============================================================
print()
print("=" * 60)
print("STEP 7: Copy Memotion images to img/ with new int-ID name")
print("=" * 60)
copied, skipped, missing = 0, 0, 0
for src, dst_name in images_to_copy:
    dst = OUT_IMG_DIR / dst_name
    if dst.exists():
        skipped += 1
        continue
    if not src.exists():
        # Try case-insensitive lookup
        parent = src.parent
        stem_l = src.stem.lower()
        ext_l = src.suffix.lower()
        found = None
        for cand in parent.iterdir():
            if cand.stem.lower() == stem_l and cand.suffix.lower() == ext_l:
                found = cand
                break
        if found is None:
            print(f"  MISSING: {src.name}")
            missing += 1
            continue
        src = found
    shutil.copy2(src, dst)
    copied += 1
print(f"  copied={copied}, skipped (already exist)={skipped}, missing={missing}")


# ============================================================
# Final summary
# ============================================================
print()
print("=" * 60)
print("DONE.")
print("=" * 60)
n_imgs = sum(1 for _ in OUT_IMG_DIR.iterdir())
print(f"merge_dataset/data/img/ now contains {n_imgs} files")
print(f"merge_dataset/data/ JSONL records: "
      f"train={len(merged['train'])}, dev={len(merged['dev'])}, test={len(merged['test'])}")
