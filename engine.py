"""Corporate Leads Pipeline engine.

Deterministic data-quality and pipeline logic. No AI calls, no network, and
no use of the system clock: "today" is always a parameter (default TODAY).

The source data is never mutated. Every fix is a *suggestion*, carried in an
issue row and, where useful, in an extra column of the flagged copy.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, utils

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TODAY = date(2026, 11, 1)  # default reference date; never read the system date

STAGES = ["New", "Contacted", "Meeting", "Proposal", "Invoice sent"]
END_STAGES = ["Paid", "Lost"]
ALL_STAGES = STAGES + END_STAGES

HARD_TYPES = (
    "overdue",
    "duplicate",
    "date_order",
    "missing_contact",
    "invalid_email",
    "stage_conflict",
)
WATCH_TYPES = ("payment_check", "going_cold", "due_today", "possible_duplicate")

ISSUE_COLUMNS = ["lead_id", "issue_type", "severity", "message", "suggested_fix"]

COLD_DAYS = 14               # going_cold: silence of at least this many days
FOLLOW_UP_OFFSET_DAYS = 3    # date_order: suggested follow-up = TODAY + 3
FUZZY_THRESHOLD = 85         # possible_duplicate: rapidfuzz WRatio cut-off

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "corporate-leads.csv"
DATE_COLUMNS = ["last_contact", "next_follow_up"]

_COLUMN_RENAMES = {"deal_value_bdt": "deal_value"}
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+$")
_PAID_BUT_MORE_DUE_RE = re.compile(r"second payment|partial|advance", re.I)
_PAYMENT_CHECK_RE = re.compile(r"advance|paid\?|not marked|check bank", re.I)


# --------------------------------------------------------------------------
# Loading and normalisation
# --------------------------------------------------------------------------
def snake_case(name: str) -> str:
    """'deal value (BDT)' -> 'deal_value_bdt', 'next follow-up' -> 'next_follow_up'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return _COLUMN_RENAMES.get(s, s)


def _blank(v) -> bool:
    return bool(pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised COPY: snake_case columns, blanks -> NaN, parsed dates.

    Idempotent, so analyze() accepts either raw or already-loaded frames.
    """
    d = df.copy()
    d.columns = [snake_case(c) for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype).startswith("string"):
            d[c] = d[c].map(
                lambda v: np.nan if _blank(v) else (v.strip() if isinstance(v, str) else v)
            )
    for c in DATE_COLUMNS:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    if "deal_value" in d.columns:
        d["deal_value"] = pd.to_numeric(d["deal_value"], errors="coerce")
    return d


def load_leads(path=DEFAULT_CSV) -> pd.DataFrame:
    """Read the CSV (BOM-safe), snake_case the columns and parse the dates."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return _prepare(raw)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _id_key(lead_id) -> tuple:
    """Natural sort key so L2 < L10; 'lowest lead id' is canonical."""
    m = re.search(r"\d+", str(lead_id))
    return (int(m.group()) if m else 0, str(lead_id))


def _email_key(v):
    return None if _blank(v) else str(v).strip().lower()


def _phone_key(v):
    if _blank(v):
        return None
    digits = re.sub(r"\D+", "", str(v))
    return digits or None


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _duplicate_map(d: pd.DataFrame) -> dict:
    """Map each non-canonical lead_id -> canonical lead_id.

    Leads are linked when they share the same non-blank email OR the same
    non-blank phone (union-find, so chains collapse into one group).
    Blank/NaN values are never compared, so blanks cannot match each other.
    """
    n = len(d)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: dict = {}
    for pos, (e, p) in enumerate(zip(d["contact_email"].map(_email_key),
                                     d["contact_phone"].map(_phone_key))):
        for kind, key in (("email", e), ("phone", p)):
            if key is None:
                continue
            k = (kind, key)
            if k in seen:
                union(pos, seen[k])
            else:
                seen[k] = pos

    groups: dict = {}
    ids = d["lead_id"].tolist()
    for pos in range(n):
        groups.setdefault(find(pos), []).append(ids[pos])

    dup_to_canon = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canon = min(members, key=_id_key)
        for m in members:
            if m != canon:
                dup_to_canon[m] = canon
    return dup_to_canon


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------
def analyze(df: pd.DataFrame, today=TODAY):
    """Return (df_with_flags, issues_df). The input frame is never modified."""
    t = pd.Timestamp(today).normalize()
    d = _prepare(df)
    rows: list[dict] = []

    def add(lead_id, issue_type, severity, message, fix):
        rows.append(dict(lead_id=lead_id, issue_type=issue_type, severity=severity,
                         message=message, suggested_fix=fix))

    is_end = d["stage"].isin(END_STAGES)
    notes = d["notes"].fillna("") if "notes" in d.columns else pd.Series("", index=d.index)

    # ---- flag columns (additive; original columns are untouched) ----------
    nf, lc = d["next_follow_up"], d["last_contact"]
    d["is_overdue"] = (nf < t) & ~is_end
    d["days_overdue"] = np.where(d["is_overdue"], (t - nf).dt.days, 0).astype(int)
    d["is_due_today"] = (nf == t) & ~is_end
    d["days_since_contact"] = (t - lc).dt.days
    d["is_going_cold"] = (d["days_since_contact"] >= COLD_DAYS) & ~is_end

    date_order = ~is_end & nf.notna() & lc.notna() & (nf < lc)
    d["corrected_follow_up"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    d.loc[date_order, "corrected_follow_up"] = t + pd.Timedelta(days=FOLLOW_UP_OFFSET_DAYS)
    effective = d["corrected_follow_up"].fillna(nf)
    d["overdue_after_fix"] = (effective < t) & ~is_end

    dup_map = _duplicate_map(d)
    d["is_duplicate"] = d["lead_id"].isin(dup_map)
    d["duplicate_of"] = d["lead_id"].map(dup_map)

    # ---- HARD 1: overdue --------------------------------------------------
    for r in d[d["is_overdue"]].itertuples():
        n = int(r.days_overdue)
        add(r.lead_id, "overdue", "high" if n >= 7 else "medium",
            f"Follow-up due {_fmt(r.next_follow_up)} is {n} day{'s' if n != 1 else ''} overdue.",
            "Contact the lead today and set a new follow-up date.")

    # ---- HARD 2: duplicate (both records get a row) -----------------------
    canon_to_dups: dict = {}
    for dup, canon in dup_map.items():
        canon_to_dups.setdefault(canon, []).append(dup)
    for dup, canon in dup_map.items():
        add(dup, "duplicate", "medium", f"Duplicate of {canon}",
            f"Exclude from totals and merge into {canon}; source row is kept as recorded.")
    for canon, dups in canon_to_dups.items():
        dups = sorted(dups, key=_id_key)
        if len(dups) == 1:
            msg = f"Surviving record; {dups[0]} is a duplicate."
        else:
            msg = f"Surviving record; {', '.join(dups)} are duplicates."
        add(canon, "duplicate", "low", msg, "Keep as the canonical record.")

    # ---- HARD 3: date_order ----------------------------------------------
    for r in d[date_order].itertuples():
        add(r.lead_id, "date_order", "medium",
            f"Next follow-up {_fmt(r.next_follow_up)} is earlier than last contact "
            f"{_fmt(r.last_contact)}.",
            "Keep last contact (a recorded event). Follow-up looks mistyped; "
            "suggest TODAY+3 pending confirmation.")

    # ---- HARD 4: missing_contact -----------------------------------------
    for r in d.itertuples():
        missing = [c for c in ("contact_name", "contact_email", "contact_phone")
                   if _blank(getattr(r, c))]
        if missing:
            add(r.lead_id, "missing_contact", "medium",
                f"Missing {', '.join(missing)}.",
                f"Collect {', '.join(missing)} from the lead or the original form.")

    # ---- HARD 5: invalid_email -------------------------------------------
    for r in d.itertuples():
        if _blank(r.contact_email):
            continue  # a blank email is missing_contact, not invalid
        email = str(r.contact_email).strip()
        domain = email.split("@", 1)[1] if "@" in email else ""
        bad_format = not _EMAIL_RE.match(email)
        upper_domain = any(ch.isupper() for ch in domain)
        if bad_format or upper_domain:
            local = email.split("@", 1)[0]
            fixed = f"{local}@{domain.lower()}" if "@" in email else email.lower()
            reason = ("uppercase letter in domain" if upper_domain and not bad_format
                      else "bad email format")
            fix = (f"Use {fixed}" if _EMAIL_RE.match(fixed)
                   else "Confirm the correct address with the contact.")
            add(r.lead_id, "invalid_email", "medium", f"Invalid email ({reason}): {email}", fix)

    # ---- HARD 6: stage_conflict (only the two defined conflicts) ----------
    for pos, r in enumerate(d.itertuples()):
        if r.stage == "New" and pd.notna(r.last_contact):
            add(r.lead_id, "stage_conflict", "medium",
                f"Stage is New but last contact is recorded ({_fmt(r.last_contact)}).",
                "Likely Contacted. Not applied, needs a human.")
        if r.stage == "Paid" and _PAID_BUT_MORE_DUE_RE.search(notes.iloc[pos]):
            add(r.lead_id, "stage_conflict", "high",
                "Stage is Paid but notes say a second/partial payment is still due.",
                "Likely Invoice sent for the balance. Not applied, needs a human.")

    # ---- WATCH 7: payment_check ------------------------------------------
    for pos, r in enumerate(d.itertuples()):
        if r.stage == "Invoice sent" and _PAYMENT_CHECK_RE.search(notes.iloc[pos]):
            add(r.lead_id, "payment_check", "low",
                "Invoice sent and notes ask to verify payment.",
                "Check the bank statement before chasing the client.")

    # ---- WATCH 8: going_cold ---------------------------------------------
    for r in d[d["is_going_cold"]].itertuples():
        add(r.lead_id, "going_cold", "low",
            f"No contact for {int(r.days_since_contact)} days (last {_fmt(r.last_contact)}).",
            "Re-engage now; 30+ days of silence means Lost per the stage rules.")

    # ---- WATCH 9: due_today ----------------------------------------------
    for r in d[d["is_due_today"]].itertuples():
        add(r.lead_id, "due_today", "low", "Follow-up is due today.",
            "Do the follow-up today.")

    # ---- WATCH 10: possible_duplicate (fuzzy company names) ---------------
    # Only leads outside every duplicate group are compared, so no pair here
    # can share an email or phone; rows with a missing company are skipped.
    cand = d[~d["is_duplicate"] & ~d["lead_id"].isin(set(dup_map.values()))
             & d["company"].notna()]
    recs = sorted(zip(cand["lead_id"], cand["company"]), key=lambda x: _id_key(x[0]))
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            (id_a, co_a), (id_b, co_b) = recs[i], recs[j]
            score = fuzz.WRatio(co_a, co_b, processor=utils.default_process)
            if score >= FUZZY_THRESHOLD:
                add(id_b, "possible_duplicate", "low",
                    f"Company '{co_b}' resembles '{co_a}' on {id_a} (similarity {score:.0f}).",
                    f"Review {id_a} and {id_b}; merge only if they are the same company.")

    issues = pd.DataFrame(rows, columns=ISSUE_COLUMNS)
    if not issues.empty:
        order = {t_: i for i, t_ in enumerate(HARD_TYPES + WATCH_TYPES)}
        issues = (issues.assign(_k=issues["lead_id"].map(_id_key),
                                _t=issues["issue_type"].map(order))
                  .sort_values(["_k", "_t"], kind="stable")
                  .drop(columns=["_k", "_t"])
                  .reset_index(drop=True))
    return d, issues


def overdue_counts(flagged: pd.DataFrame) -> dict:
    """Overdue leads as recorded vs after applying the suggested date fixes."""
    return {
        "as_recorded": int(flagged["is_overdue"].sum()),
        "after_suggested_fixes": int(flagged["overdue_after_fix"].sum()),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def clean_leads(df: pd.DataFrame) -> pd.DataFrame:
    """Rows to use in totals: duplicates excluded. The source is not touched."""
    d = _prepare(df)
    dup_map = _duplicate_map(d)
    return d[~d["lead_id"].isin(dup_map)].reset_index(drop=True)


def stage_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Lead count and total deal value per stage, in pipeline order."""
    g = clean_df.groupby("stage")["deal_value"].agg(count="size", total_deal_value="sum")
    extras = sorted(set(g.index) - set(ALL_STAGES))
    g = g.reindex(ALL_STAGES + extras, fill_value=0)
    g.index.name = "stage"
    return g.reset_index()


def scorecard(issues_df: pd.DataFrame, clean_df: pd.DataFrame | None = None) -> dict:
    """Headline numbers. Only HARD issue types are counted; WATCH ones are not."""
    hard = issues_df[issues_df["issue_type"].isin(HARD_TYPES)]
    out = {
        "leads_with_hard_issues": int(hard["lead_id"].nunique()),
        "hard_issues_by_type": hard.groupby("issue_type")["lead_id"].nunique().to_dict(),
    }
    if clean_df is not None:
        out["unique_companies"] = int(
            clean_df["company"].dropna().str.strip().str.lower().nunique()
        )
    return out