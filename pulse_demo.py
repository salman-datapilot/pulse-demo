"""
Demo 1.0 — PULSE Document Validation & Duplicate Detection
==========================================================
Single-page Streamlit interface that lets a user upload up to three documents
(CNIC front, CNIC back, electricity/utility bill), runs each through:
    - UC2 ValidatorService  : checks the document is the correct type
    - UC4 DuplicateService  : checks whether it already exists in the gallery

Per-document result:
    provided   True if a file was uploaded
    valid      True if UC2 verdict == VALID (expected class detected >= 40% conf)
    duplicated True if UC4 verdict == DUPLICATE, else False

Storage layout (created automatically):

    data/
      cnic_front/                 <- gallery: every submitted CNIC front
      cnic_back/                  <- gallery: every submitted CNIC back
      electricity_bill/           <- gallery: every submitted bill
      predictions/
        duplicated/               <- audit copy of docs called DUPLICATE
        non_duplicated/           <- audit copy of docs called NO_DUPLICATE
        inconclusive/             <- audit copy of docs called INCONCLUSIVE

Run:  streamlit run pulse_demo.py
"""

import datetime as _dt
from pathlib import Path

import streamlit as st

from doc_validator_service import ValidatorService
from dup_finder_service import DuplicateService

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

GALLERY_DIRS = {
    "cnic_front":       DATA_DIR / "cnic_front",
    "cnic_back":        DATA_DIR / "cnic_back",
    "electricity_bill": DATA_DIR / "electricity_bill",
}
PRED_DIR = DATA_DIR / "predictions"
PRED_SUBDIRS = {
    "DUPLICATE":    PRED_DIR / "duplicated",
    "NO_DUPLICATE": PRED_DIR / "non_duplicated",
    "INCONCLUSIVE": PRED_DIR / "inconclusive",
}

DOC_TYPES = {
    "cnic_front":       {"icon": "🪪", "title": "CNIC — Front"},
    "cnic_back":        {"icon": "🪪", "title": "CNIC — Back"},
    "electricity_bill": {"icon": "🧾", "title": "Utility Bill"},
}

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]


def _ensure_dirs():
    for d in GALLERY_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    for d in PRED_SUBDIRS.values():
        d.mkdir(parents=True, exist_ok=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demo 1.0 – PULSE Document Validation",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .pulse-header {
        background: linear-gradient(135deg, #1a2340 0%, #0d3b6e 100%);
        border-radius: 12px; padding: 24px 32px; margin-bottom: 28px; text-align: center;
    }
    .pulse-header h1 { color:#ffffff; font-size:1.6rem; font-weight:700; margin:0 0 4px 0; letter-spacing:0.5px; }
    .pulse-header p  { color:#8ab4d4; font-size:0.85rem; margin:0; }

    .upload-card {
        background:#f8fafc; border:1.5px solid #dde4ef; border-radius:10px;
        padding:18px 20px 14px 20px; margin-bottom:14px;
    }
    .upload-card-title { font-size:0.82rem; font-weight:600; color:#0d3b6e; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:10px; }

    .result-box {
        background:#f8fafc; border:1.5px solid #dde4ef; border-radius:10px;
        padding:18px 22px; margin-bottom:14px;
    }
    .result-section-title { font-size:0.78rem; font-weight:700; color:#0d3b6e; text-transform:uppercase; letter-spacing:0.7px; margin-bottom:10px; }
    .result-row { display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid #eef1f7; font-size:0.88rem; }
    .result-row:last-child { border-bottom:none; }
    .result-label { color:#4a5568; }
    .badge-true   { background:#d1fae5; color:#065f46; padding:2px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }
    .badge-false  { background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }
    .badge-na     { background:#e5e7eb; color:#4b5563; padding:2px 10px; border-radius:20px; font-weight:600; font-size:0.8rem; }
    .verdict-note { font-size:0.72rem; color:#9ca3af; margin-top:6px; }

    div.stButton > button {
        background: linear-gradient(135deg, #0d3b6e, #1a6eb5);
        color: white; border:none; border-radius:8px;
        padding:10px 36px; font-size:0.95rem; font-weight:600;
        letter-spacing:0.4px; width:100%; cursor:pointer; margin-top:8px;
    }
    div.stButton > button:hover { background: linear-gradient(135deg, #0a2d54, #155ea0); }
</style>
""", unsafe_allow_html=True)


# ── Services (loaded once per session) ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading validator…")
def get_validator():
    return ValidatorService(model_path=str(BASE_DIR / "best.pt"))


@st.cache_resource(show_spinner="Loading duplicate-detection service…")
def get_service():
    _ensure_dirs()
    return DuplicateService(
        galleries={k: str(v) for k, v in GALLERY_DIRS.items()},
        use_sift=True,
        use_cache=True,
        verbose=False,
    )


def reload_service():
    """Drop the cached service so newly saved gallery images are re-indexed."""
    get_service.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _timestamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def evaluate(uploaded_file, doc_type, validator, service):
    """Full per-document pipeline: validity check then duplicate check.

    Returns:
        dict(provided, valid, duplicated,
             val_verdict, val_conf,
             dup_verdict, best_match, best_score)
    """
    if uploaded_file is None:
        return {
            "provided":    False,
            "valid":       False,
            "duplicated":  False,
            "val_verdict": None,
            "val_conf":    None,
            "dup_verdict": None,
            "best_match":  None,
            "best_score":  None,
        }

    ts = _timestamp()
    safe_name = Path(uploaded_file.name).name
    raw_bytes = uploaded_file.getvalue()

    # 1) Persist to gallery folder (UC4 needs a real path; UC2 also reads it)
    gallery_path = GALLERY_DIRS[doc_type] / f"{ts}_{safe_name}"
    gallery_path.write_bytes(raw_bytes)

    # 2) UC2 — validity check
    val_res = validator.validate(str(gallery_path), doc_type)
    is_valid = val_res["valid"]

    # 3) UC4 — duplicate detection (run regardless of validity so the audit
    #    trail is complete; only shown in UI when document is valid)
    dup_res = service.find(str(gallery_path), doc_type)
    dup_verdict = dup_res["verdict"]

    # 4) Audit copy into predictions/<dup_verdict>/
    pred_dir = PRED_SUBDIRS.get(dup_verdict, PRED_SUBDIRS["INCONCLUSIVE"])
    (pred_dir / f"{doc_type}_{ts}_{safe_name}").write_bytes(raw_bytes)

    return {
        "provided":    True,
        "valid":       is_valid,
        "duplicated":  dup_verdict == "DUPLICATE",
        "val_verdict": val_res["verdict"],
        "val_conf":    val_res["best_conf"],
        "dup_verdict": dup_verdict,
        "best_match":  dup_res["best_match"],
        "best_score":  dup_res["best_score"],
    }


def badge(val):
    if val is None:
        return '<span class="badge-na">N/A</span>'
    cls = "badge-true" if val else "badge-false"
    return f'<span class="{cls}">{"True" if val else "False"}</span>'


def result_row(label, val):
    return (
        f'<div class="result-row">'
        f'<span class="result-label">{label}</span>{badge(val)}'
        f'</div>'
    )


def render_result_block(doc_type, r):
    meta = DOC_TYPES[doc_type]
    notes = []

    # UC2 note
    if r["provided"] and r["val_verdict"]:
        conf_str = f"{r['val_conf']:.2%}" if r["val_conf"] is not None else "—"
        notes.append(f"UC2 validity: {r['val_verdict']} (conf {conf_str})")

    # UC4 note — only meaningful when document is valid
    if r["provided"] and r["dup_verdict"]:
        score_str = f"{r['best_score']:.3f}" if r["best_score"] is not None else "—"
        extra = ""
        if r["dup_verdict"] == "INCONCLUSIVE":
            extra = " · routed to review"
        elif r["dup_verdict"] == "DUPLICATE" and r["best_match"]:
            extra = f" · matches {r['best_match']}"
        notes.append(f"UC4 duplicate: {r['dup_verdict']} (score {score_str}){extra}")

    notes_html = "".join(
        f'<div class="verdict-note">{n}</div>' for n in notes
    )

    st.markdown(f"""
    <div class="result-box">
        <div class="result-section-title">{meta['icon']} {meta['title']}</div>
        {result_row("Provided",   r["provided"])}
        {result_row("Valid",      r["valid"] if r["provided"] else None)}
        {result_row("Duplicated", r["duplicated"] if r["provided"] else None)}
        {notes_html}
    </div>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "results" not in st.session_state:
    st.session_state.results = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pulse-header">
    <h1>🔍 Demo 1.0</h1>
    <p>PULSE · Document Validation &amp; Duplicate Detection</p>
</div>
""", unsafe_allow_html=True)

_ensure_dirs()
validator = get_validator()
service   = get_service()

# ── Upload view ───────────────────────────────────────────────────────────────
if not st.session_state.show_results:
    st.markdown("#### Upload Documents")
    st.caption("Upload one or more documents, then click **Submit** to run validation.")

    st.markdown('<div class="upload-card"><div class="upload-card-title">🪪 CNIC — Front</div>', unsafe_allow_html=True)
    cnic_front = st.file_uploader("CNIC Front", type=IMAGE_TYPES, key="cnic_front_u", label_visibility="hidden")
    if cnic_front:
        st.image(cnic_front, caption="CNIC (front) preview", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="upload-card"><div class="upload-card-title">🪪 CNIC — Back</div>', unsafe_allow_html=True)
    cnic_back = st.file_uploader("CNIC Back", type=IMAGE_TYPES, key="cnic_back_u", label_visibility="hidden")
    if cnic_back:
        st.image(cnic_back, caption="CNIC (back) preview", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="upload-card"><div class="upload-card-title">🧾 Utility Bill</div>', unsafe_allow_html=True)
    utility = st.file_uploader("Utility Bill", type=IMAGE_TYPES, key="utility_u", label_visibility="hidden")
    if utility:
        st.image(utility, caption="Utility Bill preview", width=250)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    if st.button("Submit for Validation", use_container_width=True):
        uploads = {
            "cnic_front":       cnic_front,
            "cnic_back":        cnic_back,
            "electricity_bill": utility,
        }
        if not any(uploads.values()):
            st.warning("⚠️ Please upload at least one document before submitting.")
        else:
            with st.spinner("Running validation and duplicate detection…"):
                results = {
                    dt: evaluate(f, dt, validator, service)
                    for dt, f in uploads.items()
                }
            reload_service()
            st.session_state.results = results
            st.session_state.show_results = True
            st.rerun()

# ── Results view ──────────────────────────────────────────────────────────────
else:
    st.markdown("#### Validation Results")
    st.caption(
        "Results for the submitted documents. "
        "Valid documents have been added to the duplicate database."
    )

    r = st.session_state.results
    for dt in ("cnic_front", "cnic_back", "electricity_bill"):
        render_result_block(dt, r[dt])

    st.markdown("")
    if st.button("← Back / Upload New Documents", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.results = None
        st.rerun()