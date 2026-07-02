"""
Document Duplicate Detection Service
==========================================
A single service that, given a new document image and its document type,
decides whether the same physical document already exists in a per-type
gallery of previously submitted images.

Supported document types (each has its own gallery folder under data/):
    cnic_front
    cnic_back
    electricity_bill

    finder = DuplicateService(galleries={
        "cnic_front":       "data/cnic_front",
        "cnic_back":        "data/cnic_back",
        "electricity_bill": "data/electricity_bill",
    })
    res = finder.find(new_document="upload.jpg", document_type="cnic_front")
    print(res["verdict"])   # DUPLICATE | NO_DUPLICATE | INCONCLUSIVE

Design
------
Upstream you already classify each document, so `document_type` is known and
is used to (a) select the correct gallery folder and (b) apply the correct
calibrated decision band. The types behave differently, so a single shared
threshold is NOT used.

All types share the same matching engine: an ensemble of four geometric
similarity signals (ORB match ratio, ORB RANSAC inlier-norm, a match-volume-
gated inlier ratio, and optionally SIFT RANSAC inlier-norm), capped and
averaged into one [0,1] ensemble score per gallery image. The query's verdict
is decided from its single strongest gallery match:

    DUPLICATE      best ensemble >= HI
    NO_DUPLICATE   best ensemble <  LO
    INCONCLUSIVE   LO <= best ensemble < HI   -> route to human review

Per-type decision bands
-----------------------
Calibrated on the labeled validation runs (DUP vs HARD_NEG), tuned to maximise
the share of documents auto-decided while keeping auto-decision accuracy high:

    cnic_front        LO=0.22  HI=0.36   ~83% auto-decided @ ~98% accuracy
    cnic_back         LO=0.22  HI=0.36   (shares CNIC calibration)
    electricity_bill  LO=0.19  HI=0.53   ~47% auto-decided @ ~100% accuracy

The CNIC bands were calibrated on a combined CNIC validation set; front and
back reuse them until enough per-side labeled data exists to split them.
Bills are intrinsically harder (every bill shares the utility template and the
duplicate signal is weaker), so a larger inconclusive band is expected and
correct — those route to review (or a downstream OCR key-field check) rather
than risking a wrong auto-call. Recalibrate when a gallery grows materially.

Deps: opencv-python>=4.5, numpy
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WORK_MAX_DIM = 1024
ORB_FEATURES = 2000
SIFT_FEATURES = 1500
LOWE_RATIO = 0.75
RANSAC_REPROJ = 5.0
MIN_MATCHES_FOR_H = 8
INDEX_NAME = ".duplicate_index.pkl"

# Ensemble normalisation caps (raw signal / cap, clipped to [0,1])
CAPS = {"orb_match_ratio": 0.30,
        "orb_ransac_inlier_norm": 0.30,
        "sift_ransac_inlier_norm": 0.40,
        "gated_inlier_ratio": 0.30}

# Per-document-type decision bands (lo, hi) on the ensemble score.
# Calibrated to maximise auto-decision rate at high accuracy. See module docstring.
DECISION_BANDS = {
    "cnic_front":       {"lo": 0.22, "hi": 0.36},
    "cnic_back":        {"lo": 0.22, "hi": 0.36},
    "electricity_bill": {"lo": 0.19, "hi": 0.53},
}

# Accepted aliases -> canonical type key
TYPE_ALIASES = {
    "cnic_front": "cnic_front", "cnic-front": "cnic_front", "front": "cnic_front",
    "cnic_back": "cnic_back", "cnic-back": "cnic_back", "back": "cnic_back",
    "electricity_bill": "electricity_bill", "electricity": "electricity_bill",
    "bill": "electricity_bill", "utility_bill": "electricity_bill", "bills": "electricity_bill",
}


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def _make_sift():
    try:
        return cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    except Exception:
        return None


def _resize_max(img, max_dim=WORK_MAX_DIM):
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def extract(path, orb, sift):
    """Return a picklable dict of keypoint coordinates + descriptors."""
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(_resize_max(img), cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(gray, None)
    feat = {
        "orb_pts": np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32),
        "orb_des": des,
    }
    if sift is not None:
        skp, sdes = sift.detectAndCompute(gray, None)
        feat["sift_pts"] = np.float32([k.pt for k in skp]) if skp else np.zeros((0, 2), np.float32)
        feat["sift_des"] = sdes
    else:
        feat["sift_pts"], feat["sift_des"] = np.zeros((0, 2), np.float32), None
    return feat


# ----------------------------------------------------------------------------
# Pair scoring
# ----------------------------------------------------------------------------
def _ratio_matches(des1, des2, norm):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(des1, des2, k=2)
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance]


def _ransac_inliers(pts1, pts2, good):
    if len(good) < MIN_MATCHES_FOR_H:
        return 0
    src = pts1[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = pts2[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    return int(mask.sum()) if H is not None else 0


def pair_scores(f1, f2):
    """Compute the four raw similarity signals between two feature sets."""
    s = {}
    good = _ratio_matches(f1["orb_des"], f2["orb_des"], cv2.NORM_HAMMING)
    n_kp = min(len(f1["orb_pts"]), len(f2["orb_pts"]))
    inl = _ransac_inliers(f1["orb_pts"], f2["orb_pts"], good)

    s["orb_match_ratio"] = len(good) / n_kp if n_kp else 0.0
    s["orb_ransac_inlier_norm"] = inl / n_kp if n_kp else 0.0
    s["orb_ransac_inlier_ratio"] = inl / len(good) if good else 0.0

    if f1["sift_des"] is not None and f2["sift_des"] is not None:
        sgood = _ratio_matches(f1["sift_des"], f2["sift_des"], cv2.NORM_L2)
        sinl = _ransac_inliers(f1["sift_pts"], f2["sift_pts"], sgood)
        skp = min(len(f1["sift_pts"]), len(f2["sift_pts"]))
        s["sift_ransac_inlier_norm"] = sinl / skp if skp else 0.0
    else:
        s["sift_ransac_inlier_norm"] = None
    return s


def ensemble_score(s):
    """Fixed-weight capped mean of the signals (no learned parameters)."""
    comps = {
        "orb_match_ratio": s["orb_match_ratio"],
        "orb_ransac_inlier_norm": s["orb_ransac_inlier_norm"],
        "gated_inlier_ratio": s["orb_ransac_inlier_ratio"] * s["orb_match_ratio"],
    }
    if s["sift_ransac_inlier_norm"] is not None:
        comps["sift_ransac_inlier_norm"] = s["sift_ransac_inlier_norm"]
    vals = [min(max(v / CAPS[k], 0.0), 1.0) for k, v in comps.items()]
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# Per-type gallery (feature index with on-disk cache)
# ----------------------------------------------------------------------------
class _Gallery:
    def __init__(self, root, orb, sift, use_cache=True, verbose=True):
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"Gallery root not found: {root}")
        self.orb, self.sift, self.verbose = orb, sift, verbose
        self.index = self._build_index(use_cache)

    def _files(self):
        return sorted(p for p in self.root.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)

    def _build_index(self, use_cache):
        cache_path = self.root / INDEX_NAME
        cached = {}
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)
            except Exception:
                cached = {}

        index, dirty = {}, False
        for p in self._files():
            key = str(p.relative_to(self.root))
            mtime = p.stat().st_mtime
            entry = cached.get(key)
            sift_ok = (entry or {}).get("has_sift") == (self.sift is not None)
            if entry and entry["mtime"] == mtime and sift_ok:
                index[key] = entry
                continue
            if self.verbose:
                print(f"  indexing: {self.root.name}/{key}")
            index[key] = {"mtime": mtime,
                          "feat": extract(p, self.orb, self.sift),
                          "has_sift": self.sift is not None}
            dirty = True

        if use_cache and (dirty or set(index) != set(cached)):
            try:
                with open(cache_path, "wb") as fh:
                    pickle.dump(index, fh)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] could not write cache for {self.root}: {e}")
        if self.verbose:
            print(f"Gallery '{self.root.name}': {len(index)} images "
                  f"({'cached' if not dirty else 'updated'})")
        return index


# ----------------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------------
class DuplicateService:
    """Multi-type duplicate detector. Galleries and decision bands are keyed by
    canonical document type ('cnic_front', 'cnic_back', 'electricity_bill')."""

    def __init__(self, galleries, use_sift=True, bands=None,
                 use_cache=True, verbose=True):
        """
        galleries : dict {document_type: gallery_folder_path}
        bands     : optional override {document_type: {"lo":..,"hi":..}}
        """
        self.verbose = verbose
        self.bands = {**DECISION_BANDS, **(bands or {})}
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.sift = _make_sift() if use_sift else None
        if use_sift and self.sift is None and verbose:
            print("[warn] SIFT unavailable; ensemble uses 3 signals")

        self.galleries = {}
        for dtype, root in galleries.items():
            key = self._canon(dtype)
            self.galleries[key] = _Gallery(root, self.orb, self.sift,
                                           use_cache=use_cache, verbose=verbose)

    @staticmethod
    def _canon(document_type):
        key = TYPE_ALIASES.get(str(document_type).strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown document_type '{document_type}'. "
                f"Expected one of: {sorted(set(TYPE_ALIASES.values()))}")
        return key

    def find(self, new_document, document_type, topk=5):
        """Compare `new_document` against the gallery for `document_type`.

        Returns dict:
            verdict       'DUPLICATE' | 'NO_DUPLICATE' | 'INCONCLUSIVE'
            document_type canonical type used
            best_match    gallery-relative path of strongest candidate (or None)
            best_score    ensemble score of that candidate
            candidates    top-k [{path, ensemble, <raw signals>}]
            thresholds    (lo, hi) applied
        """
        dtype = self._canon(document_type)
        if dtype not in self.galleries:
            raise ValueError(f"No gallery configured for document_type '{dtype}'")
        gallery = self.galleries[dtype]
        band = self.bands[dtype]
        lo, hi = band["lo"], band["hi"]

        qpath = Path(new_document)
        qfeat = extract(qpath, self.orb, self.sift)

        rows = []
        for key, entry in gallery.index.items():
            if qpath.resolve() == (gallery.root / key).resolve():
                continue  # never match an image against itself
            raw = pair_scores(qfeat, entry["feat"])
            rows.append({"path": key, "ensemble": ensemble_score(raw), **raw})

        if not rows:
            return {"verdict": "NO_DUPLICATE", "document_type": dtype,
                    "best_match": None, "best_score": 0.0,
                    "candidates": [], "thresholds": (lo, hi)}

        rows.sort(key=lambda r: r["ensemble"], reverse=True)
        best = rows[0]
        if best["ensemble"] >= hi:
            verdict = "DUPLICATE"
        elif best["ensemble"] < lo:
            verdict = "NO_DUPLICATE"
        else:
            verdict = "INCONCLUSIVE"

        return {"verdict": verdict, "document_type": dtype,
                "best_match": best["path"],
                "best_score": round(best["ensemble"], 4),
                "candidates": rows[:topk], "thresholds": (lo, hi)}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Detect duplicates of a new document against a per-type gallery")
    ap.add_argument("new_document", help="path to the incoming image")
    ap.add_argument("document_type", help="cnic_front | cnic_back | electricity_bill")
    ap.add_argument("--data-root", default="data",
                    help="parent folder containing the per-type gallery folders")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lo", type=float, default=None, help="override LO band")
    ap.add_argument("--hi", type=float, default=None, help="override HI band")
    ap.add_argument("--no-sift", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    galleries = {
        "cnic_front":       str(data_root / "cnic_front"),
        "cnic_back":        str(data_root / "cnic_back"),
        "electricity_bill": str(data_root / "electricity_bill"),
    }

    bands = None
    if args.lo is not None or args.hi is not None:
        dt = DuplicateService._canon(args.document_type)
        base = DECISION_BANDS[dt]
        bands = {dt: {"lo": args.lo if args.lo is not None else base["lo"],
                      "hi": args.hi if args.hi is not None else base["hi"]}}

    svc = DuplicateService(galleries=galleries, use_sift=not args.no_sift,
                           bands=bands, use_cache=not args.no_cache)
    res = svc.find(args.new_document, args.document_type, topk=args.topk)

    lo, hi = res["thresholds"]
    print(f"\nQuery:     {args.new_document}")
    print(f"Type:      {res['document_type']}")
    print(f"Verdict:   {res['verdict']}   "
          f"(score {res['best_score']:.3f} | band: <{lo} no-dup, >={hi} dup)")
    if res["best_match"]:
        print(f"Best match: {res['best_match']}")
    if res["candidates"]:
        print(f"\nTop {len(res['candidates'])} candidates:")
        hdr = (f"{'gallery image':35s} {'ens':>6s} {'orb_mr':>7s} "
               f"{'orb_in':>7s} {'ratio':>6s} {'sift':>6s}")
        print(hdr); print("-" * len(hdr))
        for c in res["candidates"]:
            sift = (f"{c['sift_ransac_inlier_norm']:.3f}"
                    if c["sift_ransac_inlier_norm"] is not None else "  n/a")
            print(f"{c['path']:35s} {c['ensemble']:6.3f} {c['orb_match_ratio']:7.3f} "
                  f"{c['orb_ransac_inlier_norm']:7.3f} {c['orb_ransac_inlier_ratio']:6.3f} {sift:>6s}")
    if res["verdict"] == "INCONCLUSIVE":
        print("\n-> route to human review (or OCR key-field check for bills).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
