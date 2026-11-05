"""Shared UI helpers for every screen: palette, CSS, cached data, formatting."""
from __future__ import annotations

import html
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import engine

# --------------------------------------------------------------------------
# Palette (one place, used by CSS, badges and stage colours)
# --------------------------------------------------------------------------
INK = "#111827"
MUTED = "#6B7280"
LINE = "#E5E7EB"
SURFACE = "#FFFFFF"
CANVAS = "#F5F7FB"
PRIMARY = "#4F46E5"      # indigo
RED = "#DC2626"
AMBER = "#D97706"
ORANGE = "#EA580C"
BLUE = "#2563EB"
GREY = "#6B7280"
GREEN = "#059669"

STAGE_COLORS = {
    "New": "#94A3B8",
    "Contacted": "#38BDF8",
    "Meeting": "#818CF8",
    "Proposal": "#A78BFA",
    "Invoice sent": "#F59E0B",
    "Paid": GREEN,
    "Lost": "#9CA3AF",
}

ISSUE_LABELS = {
    "overdue": "Overdue follow-up",
    "duplicate": "Duplicate",
    "date_order": "Date order error",
    "missing_contact": "Missing contact details",
    "invalid_email": "Invalid email",
    "stage_conflict": "Stage conflict",
    "payment_check": "Payment check",
    "going_cold": "Going cold",
    "due_today": "Due today",
    "possible_duplicate": "Possible duplicate",
}

CSS = f"""
<style>
:root {{
  --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --surface:{SURFACE}; --canvas:{CANVAS};
  --primary:{PRIMARY}; --red:{RED}; --amber:{AMBER}; --orange:{ORANGE}; --blue:{BLUE}; --grey:{GREY};
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 100%; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: var(--canvas); }}
div[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}}
div[data-testid="stMetricLabel"] p {{ color: var(--muted); font-size: .8rem; }}
.scorecard {{
  background: linear-gradient(90deg, #EEF2FF, #F8FAFF); border: 1px solid #E0E7FF;
  color: #312E81; border-radius: 12px; padding: 12px 18px; font-weight: 600;
  font-size: .98rem; margin: 4px 0 18px 0;
}}
.kh {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 10px 12px; margin-bottom: 10px; border-top-width: 4px; }}
.kh-top {{ display:flex; justify-content:space-between; align-items:center; }}
.kh-name {{ font-weight: 700; font-size: .9rem; color: var(--ink); }}
.kh-count {{ background: var(--canvas); color: var(--ink); border-radius: 999px;
  padding: 1px 9px; font-size: .78rem; font-weight: 700; }}
.kh-val {{ color: var(--muted); font-size: .78rem; margin-top: 2px; }}
.kh-note {{ color: var(--muted); font-size: .7rem; }}
.lc {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 10px 12px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(16,24,40,.05); }}
.lc.dup {{ background: #F9FAFB; opacity: .72; border-style: dashed; }}
.lc-co {{ font-weight: 700; font-size: .88rem; color: var(--ink); line-height: 1.25; }}
.lc-id {{ color: var(--muted); font-weight: 500; font-size: .72rem; margin-left: 4px; }}
.lc-sub {{ color: var(--muted); font-size: .74rem; margin-top: 2px; }}
.lc-val {{ color: var(--primary); font-weight: 700; font-size: .84rem; margin-top: 6px; }}
.lc-fu {{ color: var(--muted); font-size: .72rem; margin-top: 2px; }}
.bdgs {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:8px; align-items:center; }}
.bdg {{ border-radius: 999px; padding: 1px 8px; font-size: .68rem; font-weight: 700; white-space: nowrap; }}
.bdg.red {{ background:#FEE2E2; color:#B91C1C; }}
.bdg.amber {{ background:#FEF3C7; color:#B45309; }}
.bdg.grey {{ background:#E5E7EB; color:#4B5563; }}
.bdg.orange {{ background:#FFEDD5; color:#C2410C; }}
.bdg.blue {{ background:#DBEAFE; color:#1D4ED8; }}
.bdg.green {{ background:#D1FAE5; color:#047857; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; background: var(--blue); }}
.legend {{ color: var(--muted); font-size: .78rem; margin: 2px 0 14px 0; }}
.kv-l {{ color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing:.04em; }}
.kv-v {{ color: var(--ink); font-size: .95rem; margin-bottom: 12px; word-break: break-word; }}
.kv-missing {{ color: var(--red); font-weight: 600; }}
.issue {{ border:1px solid var(--line); border-left-width:4px; border-radius:10px;
  padding:10px 14px; margin-bottom:10px; background: var(--surface); }}
.issue.hard {{ border-left-color: var(--orange); }}
.issue.watch {{ border-left-color: var(--blue); }}
.issue-h {{ font-weight: 700; color: var(--ink); font-size: .9rem; }}
.issue-m {{ color: var(--ink); font-size: .88rem; margin-top: 4px; }}
.issue-f {{ color: var(--muted); font-size: .84rem; margin-top: 4px; }}
.hold {{ background:#FFFBEB; border:1px solid #FDE68A; border-left:5px solid var(--amber);
  border-radius:12px; padding:14px 16px; color:#92400E; margin-bottom:10px; }}
.hold b {{ color:#78350F; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Reference date + cached analysis
# --------------------------------------------------------------------------
def get_today() -> date:
    """The sidebar reference date. Pages fall back to the brief's date if run alone."""
    v = st.session_state.get("today_ref")
    return v if isinstance(v, date) else engine.TODAY


@st.cache_data(show_spinner=False)
def _analysis(today: date):
    src = engine.load_leads()
    flagged, issues = engine.analyze(src, today)
    clean = engine.clean_leads(src)
    return flagged, issues, clean


def get_data():
    """(flagged, issues, clean) for the current reference date."""
    return _analysis(get_today())


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def esc(v) -> str:
    return html.escape("" if v is None else str(v)).replace("$", "&#36;")


def blank(v) -> bool:
    return bool(pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def fmt_bdt(v) -> str:
    return "BDT 0" if blank(v) else f"BDT {float(v):,.0f}"


def fmt_date(v, empty="—") -> str:
    return empty if blank(v) else pd.Timestamp(v).strftime("%d %b %Y")


def plural(n: int, word: str, plural_word: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural_word or word + 's')}"


def to_py(v):
    """numpy / pandas scalars -> plain Python (blank -> None, dates -> ISO strings)."""
    if blank(v):
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, np.generic):
        return v.item()
    return v


def lead_to_dict(row: pd.Series, issues: pd.DataFrame) -> dict:
    """Lead as a plain dict for the AI layer (see ai.py for the contract)."""
    d = {k: to_py(v) for k, v in row.items()}
    mine = issues[issues["lead_id"] == row["lead_id"]]
    d["issues"] = mine.to_dict("records")
    return d


def scorecard_line(flagged, issues, clean) -> str:
    sc = engine.scorecard(issues, clean)
    oc = engine.overdue_counts(flagged)
    dups = int(flagged["is_duplicate"].sum())
    dates = int((issues["issue_type"] == "date_order").sum())
    return (
        f"{sc['unique_companies']} unique companies · {plural(dups, 'duplicate')} caught · "
        f"{plural(dates, 'date error')} · {oc['as_recorded']} overdue "
        f"({oc['after_suggested_fixes']} after fix) · "
        f"{sc['leads_with_hard_issues']} of {len(flagged)} leads have a hard data issue"
    )


def issue_tier(issue_type: str) -> str:
    return "Hard" if issue_type in engine.HARD_TYPES else "Watch"
