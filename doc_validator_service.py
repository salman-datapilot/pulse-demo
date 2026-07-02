"""
UC2 — Document Validity Service
================================
Given a document image and its expected document type, decides whether the
image is a valid instance of that type using a YOLO object-detection model.

Supported document types:
    cnic_front
    cnic_back
    electricity_bill

The model was trained on five classes:
    0: cnic_back
    1: cnic_front
    2: invalid_bill
    3: invalid_cnic
    4: utility_bill

Validity rule: a document is VALID if the model detects the expected class
for that slot with confidence >= conf_threshold (default 0.40), regardless
of any other classes also detected in the same image.

    from doc_validator_service import ValidatorService

    validator = ValidatorService(model_path="best.pt")
    res = validator.validate(image_path="upload.jpg", document_type="cnic_front")
    print(res["valid"])    # True | False
    print(res["verdict"])  # VALID | INVALID | NO_DETECTION

Design
------
Only the detection for the expected class is considered. If the model also
fires on invalid_cnic or invalid_bill, those detections are ignored — the
question is purely "did the model see the right document type here at
sufficient confidence?"

Per-slot expected classes
--------------------------
    cnic_front       -> model class 'cnic_front'
    cnic_back        -> model class 'cnic_back'
    electricity_bill -> model class 'utility_bill'

Deps: ultralytics, torch (via ultralytics)
"""

from pathlib import Path

from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "best.pt"
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.45

# Maps the app's slot name -> the YOLO class name that must be detected
EXPECTED_CLASS = {
    "cnic_front":       "cnic_front",
    "cnic_back":        "cnic_back",
    "electricity_bill": "utility_bill",
}

# Accepted aliases -> canonical slot key (mirrors dup_finder_service convention)
TYPE_ALIASES = {
    "cnic_front":       "cnic_front",
    "cnic-front":       "cnic_front",
    "front":            "cnic_front",
    "cnic_back":        "cnic_back",
    "cnic-back":        "cnic_back",
    "back":             "cnic_back",
    "electricity_bill": "electricity_bill",
    "electricity":      "electricity_bill",
    "bill":             "electricity_bill",
    "utility_bill":     "electricity_bill",
    "bills":            "electricity_bill",
}


class ValidatorService:
    """
    Wraps a YOLO model to validate uploaded document images against their
    expected document type.

    Parameters
    ----------
    model_path       : path to best.pt (defaults to same folder as this file)
    conf_threshold   : minimum confidence to count a detection (default 0.40)
    iou_threshold    : NMS IoU threshold passed to YOLO (default 0.45)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path.resolve()}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(str(model_path))

        # Build reverse lookup: class_name -> class_id (for reference / logging)
        self.class_names: dict[int, str] = self.model.names  # {id: name}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _canon(document_type: str) -> str:
        key = TYPE_ALIASES.get(str(document_type).strip().lower())
        if key is None:
            raise ValueError(
                f"Unknown document_type '{document_type}'. "
                f"Expected one of: {sorted(set(TYPE_ALIASES.values()))}"
            )
        return key

    def _detect(self, image_path: str) -> list[dict]:
        """Run YOLO inference and return all detections as a list of dicts."""
        results = self.model.predict(
            source=image_path,
            task="detect",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        result = results[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append({
                    "class_id":   class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2),
                    },
                })
        return detections

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        image_path: str,
        document_type: str,
    ) -> dict:
        """
        Validate a single document image against its expected type.

        Parameters
        ----------
        image_path    : path to the image file
        document_type : slot name ('cnic_front', 'cnic_back', 'electricity_bill')

        Returns
        -------
        dict with keys:
            valid           bool   — True if the expected class was detected >= threshold
            verdict         str    — 'VALID' | 'INVALID' | 'NO_DETECTION'
            document_type   str    — canonical slot name
            expected_class  str    — YOLO class name that was required
            best_conf       float  — highest confidence for the expected class (0.0 if absent)
            conf_threshold  float  — threshold applied
            all_detections  list   — full list of every detection from the model
        """
        dtype = self._canon(document_type)
        expected = EXPECTED_CLASS[dtype]

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        all_detections = self._detect(image_path)

        # Filter to detections of the expected class only
        expected_detections = [
            d for d in all_detections if d["class_name"] == expected
        ]

        if not all_detections:
            verdict = "NO_DETECTION"
            best_conf = 0.0
        elif not expected_detections:
            # Model fired, but not on the expected class
            verdict = "INVALID"
            best_conf = 0.0
        else:
            # Take highest-confidence hit for the expected class
            best_conf = max(d["confidence"] for d in expected_detections)
            verdict = "VALID" if best_conf >= self.conf_threshold else "INVALID"

        return {
            "valid":          verdict == "VALID",
            "verdict":        verdict,
            "document_type":  dtype,
            "expected_class": expected,
            "best_conf":      best_conf,
            "conf_threshold": self.conf_threshold,
            "all_detections": all_detections,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Validate a document image against its expected type."
    )
    ap.add_argument("image_path",     help="Path to the document image")
    ap.add_argument("document_type",  help="cnic_front | cnic_back | electricity_bill")
    ap.add_argument("--model",        default=DEFAULT_MODEL_PATH, help="Path to best.pt")
    ap.add_argument("--conf",         type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold (default 0.40)")
    ap.add_argument("--iou",          type=float, default=DEFAULT_IOU_THRESHOLD,  help="IoU threshold (default 0.45)")
    args = ap.parse_args()

    svc = ValidatorService(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    res = svc.validate(args.image_path, args.document_type)

    print(f"\nImage:          {args.image_path}")
    print(f"Slot:           {res['document_type']}")
    print(f"Expected class: {res['expected_class']}")
    print(f"Verdict:        {res['verdict']}  (best_conf={res['best_conf']:.4f}, threshold={res['conf_threshold']})")
    print(f"\nAll detections ({len(res['all_detections'])}):")
    for d in res["all_detections"]:
        marker = " <-- expected" if d["class_name"] == res["expected_class"] else ""
        print(f"  {d['class_name']:20s}  conf={d['confidence']:.4f}{marker}")

    sys.exit(0 if res["valid"] else 1)