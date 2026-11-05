"""AI layer: OpenAI Python SDK pointed at Groq (primary model: openai/gpt-oss-120b).

Division of labour: the model only (1) writes emails, (2) turns a question
into a JSON filter spec, (3) words an answer from numbers we computed, and
(4) writes the weekly "chase first" list from facts we computed. Every number
and every choice of WHICH rows to show comes from pandas / engine.py, never
from the model. Deterministic guards (Lost, duplicate, no email, "check bank")
return HOLD without calling the model at all.

Key lookup: st.secrets["GROQ_API_KEY"] if available, else the environment /
.env file (python-dotenv). Calls use a 2-second backoff and one retry; the
retry goes to the fallback model. On failure each function returns a friendly
fallback instead of raising.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

import engine

# Confirm both names against Groq's current model list before recording the demo.
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]   # primary, fallback (used on the retry)
MODEL = MODELS[0]
BASE_URL = "https://api.groq.com/openai/v1"
BACKOFF_SECONDS = 2
MAX_ATTEMPTS = 2          # first try + 1 retry (on the fallback model)
TIMEOUT_SECONDS = 45

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
EMAIL_SYSTEM_PROMPT = """You write short follow-up emails for BIAI, a Bangladeshi AI training company, to corporate leads. Polite, warm, professional English, 70-120 words. Structure: greeting with the contact's name; one line referring to the specific last interaction in the notes; one clear next step with a suggested action (not a vague 'let me know'); polite close signed 'BIAI Team'. Rules: use ONLY facts in the lead data and notes; never invent prices, dates or promises. Stage Invoice sent: be gentle, ask whether they need anything from us to complete payment, never demand it. Stage Proposal: offer a quick call to walk through it. Stage Meeting: offer next steps toward a proposal. Stage Paid: a light note is fine, reference training kickoff, never request payment. HOLD rule: if the notes say the lead is lost or 'do not contact', or say to check internal records (for example 'check bank') before chasing, do NOT write an email; output one line starting with 'HOLD:' explaining why. If the contact name is missing, use 'Hello' and add a final line starting with 'NOTE:' saying the contact name and email must be confirmed. Output 'Subject: ...' first, then the body. No markdown. Never state dates (e.g. 'sent on X') or program details unless they are explicitly written in the notes. If the notes say 'Ask whether the advance has come in', you MUST explicitly ask that question in the body."""

QUESTION_PARSER_SYSTEM_PROMPT = """You convert a user's question about a sales pipeline into the JSON schema provided. Stages: New, Contacted, Meeting, Proposal, Invoice sent, Paid, Lost. 'Overdue' means the follow-up date is before the reference date. 'Stuck' or 'chase first' means intent "rank". The user may write in English, Bangla or Banglish. Do not answer the question. Output JSON only."""

ANSWER_WRITER_SYSTEM_PROMPT = """You are a pipeline assistant. Answer in 1-2 plain sentences using ONLY the numbers and rows provided. Start with the number of matching leads (metrics.matched_leads). Currency is BDT, formatted with commas. Never compute or guess numbers yourself."""

RANK_WRITER_SYSTEM_PROMPT = """You are a pipeline assistant. The deals provided are already ranked; list them in exactly that order, one line each: lead id, company, deal value (say 'no agreed value' if it is BDT 0), days overdue, and the caveat if one is given. Use ONLY the rows provided. Do not add a heading or a count. Never compute or guess numbers yourself."""

WEEKLY_SYSTEM_PROMPT = (
    "You write BIAI's weekly pipeline summary. Use ONLY the facts provided; never add claims or judge importance. "
    "Write a numbered list starting at 1 of the 'chase_first' deals in the order given, one line each: lead id, company, "
    "deal value (say 'no agreed value' if it is BDT 0), days overdue, and a next action taken from its notes. "
    "If a deal has a caveat, include it; never call a deal 'unpaid' when its caveat says payment is unverified. "
    "Maximum 150 words. Do not mention data problems; the system adds that line separately."
)

PARSER_SCHEMA = """Return ONLY a JSON object with exactly this shape:
{"intent": "list"|"count"|"sum"|"rank"|"unsupported",
 "filters": {"stage": [..] | null, "overdue": true|false|null, "industry": [..]|null,
             "has_issue": true|false|null, "missing_contact": true|false|null, "min_value": number|null},
 "sort_by": "deal_value"|"days_overdue"|"next_follow_up"|null,
 "limit": number|null}
Use null for any filter the question does not mention. Use "unsupported" when the question cannot be
answered by filtering, counting, summing or ranking leads."""

# --------------------------------------------------------------------------
# Friendly fallbacks
# --------------------------------------------------------------------------
FALLBACK_DRAFT = ("[Draft unavailable] The AI service didn't respond, so no email was written. "
                  "Please try again in a moment. Nothing here should be sent.")
FALLBACK_ANSWER = ("Sorry, I couldn't reach the AI service just now, so I can't answer that. "
                   "Please try again in a moment.")
FALLBACK_SUMMARY = ("The weekly summary couldn't be generated because the AI service didn't respond. "
                    "Please try again in a moment.")
NOT_CONFIGURED_HINT = " (No API key found: add GROQ_API_KEY to .streamlit/secrets.toml or to a .env file.)"
UNSUPPORTED_TEXT = (
    "I can't answer that one yet. I can list, count, sum or rank leads by stage, overdue status, "
    "industry, data issues, missing contact details or minimum deal value. For example: "
    "\"Which deals are overdue?\" or \"What is the total value in Proposal stage?\""
)

# "Chase first" questions are ranked by code, whatever intent the model guessed.
_CHASE_RE = re.compile(r"chase|prioriti[sz]e|\bstuck\b|\burgent\b", re.I)
# Notes that tell a human to check internal records before any chasing email.
_VERIFY_FIRST_RE = re.compile(r"check bank|paid\?\s*not marked", re.I)


class AIUnavailable(Exception):
    """The model could not be reached or gave nothing usable."""


class AINotConfigured(AIUnavailable):
    """No API key (or no openai package). Retrying would not help."""


# --------------------------------------------------------------------------
# Client + call wrapper
# --------------------------------------------------------------------------
_client = None
_client_key = None


def _api_key() -> str | None:
    try:  # Streamlit secrets, when running in the app and a secrets file exists
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    try:  # .env next to this file (never overrides a real environment variable)
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or None


def _get_client():
    global _client, _client_key
    key = _api_key()
    if not key:
        raise AINotConfigured("GROQ_API_KEY not set")
    if _client is None or key != _client_key:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise AINotConfigured("openai package not installed") from e
        # max_retries=0: the retry policy lives here (one retry, 2 s backoff), not in the SDK.
        _client = OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS, max_retries=0)
        _client_key = key
    return _client


def _call_llm(system: str, user: str, *, json_mode: bool = False,
              temperature: float = 0.3, max_tokens: int = 2000) -> str:
    """One chat completion with a 2 s backoff and a single retry on the fallback model."""
    client = _get_client()
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        model = MODELS[min(attempt, len(MODELS) - 1)]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise AIUnavailable("empty response")
            return text
        except Exception as e:  # noqa: BLE001 - any failure is retried once
            last = e
            if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                break  # a retry cannot fix a bad key
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS)
    raise AIUnavailable(str(last)) from last


def _hint(e: Exception | None) -> str:
    return NOT_CONFIGURED_HINT if isinstance(e, AINotConfigured) else ""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _bdt(v) -> str:
    return "BDT 0" if v is None or pd.isna(v) else f"BDT {float(v):,.0f}"


def _s(v):
    """JSON-safe scalar: NaN/NaT -> None, timestamps -> 'YYYY-MM-DD', numpy -> python."""
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "item"):
        return v.item()
    return v


def _loads_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise
        out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("not a JSON object")
    return out


def _infer_today(df: pd.DataFrame):
    """Recover the reference date a flagged frame was analysed with
    (exact: last_contact + days_since_contact). Falls back to the brief's date."""
    if {"last_contact", "days_since_contact"} <= set(df.columns):
        ok = df["last_contact"].notna() & df["days_since_contact"].notna()
        if ok.any():
            r = df[ok].iloc[0]
            return (pd.Timestamp(r["last_contact"]) + pd.Timedelta(days=int(r["days_since_contact"]))).date()
    return engine.TODAY


def _frames(df: pd.DataFrame):
    """(flagged, issues, live): re-analysed at the frame's own date; live = de-duplicated rows."""
    flagged, issues = engine.analyze(df, _infer_today(df))
    live = flagged[~flagged["is_duplicate"]].reset_index(drop=True)
    return flagged, issues, live


# --------------------------------------------------------------------------
# 1. Draft follow-up email
# --------------------------------------------------------------------------
def draft_email(lead: dict) -> str:
    # Deterministic guards (no model call): never draft for these.
    if lead.get("stage") == "Lost":
        return ("HOLD: This lead is marked Lost, so no follow-up should go out. "
                f"Notes say: {lead.get('notes') or 'no notes'}")
    if lead.get("is_duplicate"):
        return (f"HOLD: This record is a duplicate of {lead.get('duplicate_of')}. "
                "Email the surviving record instead so the contact is not messaged twice.")

    email = lead.get("contact_email")
    if email is None or pd.isna(email) or str(email).strip() == "":
        phone = lead.get("contact_phone")
        phone = "no phone on file" if phone is None or pd.isna(phone) or not str(phone).strip() else phone
        return f"HOLD: No email address on file. Call {phone} instead."

    notes = lead.get("notes")
    notes = "" if notes is None or pd.isna(notes) else str(notes)
    if _VERIFY_FIRST_RE.search(notes):
        return ("HOLD: The notes say to check the bank / internal records before chasing, "
                f"so no email should go out yet. Notes say: {notes}")

    hard_types = lead.get("hard_issue_types")
    if hard_types is None:
        hard_types = sorted({i.get("issue_type") for i in lead.get("issues", [])
                             if i.get("issue_type") in engine.HARD_TYPES})
    payload = {k: _s(lead.get(k)) for k in (
        "company", "contact_name", "contact_role", "stage", "deal_value",
        "last_contact", "next_follow_up", "notes")}
    payload["hard_issue_types"] = list(hard_types)
    try:
        return _call_llm(EMAIL_SYSTEM_PROMPT,
                         "Lead data (JSON):\n" + json.dumps(payload, ensure_ascii=False),
                         temperature=0.4)
    except AIUnavailable as e:
        return FALLBACK_DRAFT + _hint(e)


# --------------------------------------------------------------------------
# 2. Question -> JSON filter spec
# --------------------------------------------------------------------------
_STAGE_LOOKUP = {s.lower(): s for s in engine.ALL_STAGES} | {"invoice": "Invoice sent"}
_INTENTS = {"list", "count", "sum", "rank", "unsupported"}
_SORTS = {"deal_value", "days_overdue", "next_follow_up"}


def _empty_spec(intent: str = "unsupported") -> dict:
    return {"intent": intent,
            "filters": {"stage": None, "overdue": None, "industry": None,
                        "has_issue": None, "missing_contact": None, "min_value": None},
            "sort_by": None, "limit": None}


def _as_list(v):
    if v is None:
        return None
    items = v if isinstance(v, list) else [v]
    items = [str(x).strip() for x in items if str(x).strip()]
    return items or None


def _as_bool(v):
    return v if isinstance(v, bool) else None


def _normalize_spec(raw: dict) -> dict:
    """Coerce the model's JSON into the schema; anything off-schema becomes null."""
    spec = _empty_spec(raw.get("intent") if raw.get("intent") in _INTENTS else "unsupported")
    f = raw.get("filters") if isinstance(raw.get("filters"), dict) else {}
    stages = _as_list(f.get("stage"))
    if stages:
        mapped = [_STAGE_LOOKUP.get(re.sub(r"[_\-]+", " ", s).lower()) for s in stages]
        spec["filters"]["stage"] = [m for m in mapped if m] or None
    spec["filters"]["overdue"] = _as_bool(f.get("overdue"))
    spec["filters"]["industry"] = _as_list(f.get("industry"))
    spec["filters"]["has_issue"] = _as_bool(f.get("has_issue"))
    spec["filters"]["missing_contact"] = _as_bool(f.get("missing_contact"))
    mv = f.get("min_value")
    spec["filters"]["min_value"] = mv if isinstance(mv, (int, float)) and not isinstance(mv, bool) else None
    spec["sort_by"] = raw.get("sort_by") if raw.get("sort_by") in _SORTS else None
    lim = raw.get("limit")
    spec["limit"] = int(lim) if isinstance(lim, (int, float)) and not isinstance(lim, bool) and lim >= 1 else None
    return spec


def _parse(question: str) -> tuple[dict, Exception | None]:
    try:
        text = _call_llm(QUESTION_PARSER_SYSTEM_PROMPT,
                         f"{PARSER_SCHEMA}\n\nQuestion: {question}",
                         json_mode=True, temperature=0.0, max_tokens=1500)
        return _normalize_spec(_loads_json(text)), None
    except (AIUnavailable, ValueError) as e:   # JSONDecodeError is a ValueError
        return _empty_spec(), e


def parse_question(question: str) -> dict:
    """Question -> {"intent", "filters", "sort_by", "limit"}.

    If the AI call fails, returns an "unsupported" spec plus an extra "error" key.
    """
    spec, err = _parse(question)
    if err is not None:
        spec["error"] = str(err)
    return spec


# --------------------------------------------------------------------------
# 3. Apply the spec in pandas, then let the model word the result
# --------------------------------------------------------------------------
def apply_filters(live: pd.DataFrame, issues: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """All the maths happens here. `live` must already be de-duplicated."""
    f, out = spec["filters"], live
    if f["stage"]:
        out = out[out["stage"].isin(f["stage"])]
    if f["overdue"] is not None:
        out = out[out["is_overdue"] == f["overdue"]]
    if f["industry"]:
        wanted = [w.lower() for w in f["industry"]]
        ind = out["industry"].fillna("").str.lower()
        out = out[ind.map(lambda x: any(w == x or w in x for w in wanted))]
    if f["has_issue"] is not None:
        hard_ids = set(issues.loc[issues["issue_type"].isin(engine.HARD_TYPES), "lead_id"])
        out = out[out["lead_id"].isin(hard_ids) == f["has_issue"]]
    if f["missing_contact"] is not None:
        gone = pd.Series(False, index=out.index)
        for c in ("contact_name", "contact_email", "contact_phone"):
            gone |= out[c].isna() | (out[c].astype(str).str.strip() == "")
        out = out[gone == f["missing_contact"]]
    if f["min_value"] is not None:
        out = out[out["deal_value"].fillna(0) >= f["min_value"]]

    sort_by = spec["sort_by"]
    if sort_by == "days_overdue":
        out = out.sort_values(["days_overdue", "deal_value", "lead_id"], ascending=[False, False, True])
    elif sort_by == "deal_value":
        out = out.sort_values(["deal_value", "days_overdue", "lead_id"], ascending=[False, False, True])
    elif sort_by == "next_follow_up":
        out = out.sort_values(["next_follow_up", "lead_id"], na_position="last")
    if spec["limit"]:
        out = out.head(spec["limit"])
    return out.reset_index(drop=True)


def _caveats(issues: pd.DataFrame) -> dict:
    """lead_id -> short warning, derived from the engine's issue rows (nothing hardcoded)."""
    texts = {"payment_check": "verify payment first (status unverified)",
             "date_order": "date suspect (follow-up is before last contact)",
             "invalid_email": "fix the email address first",
             "missing_contact": "contact details missing"}
    out: dict = {}
    for r in issues.itertuples():
        if r.issue_type in texts:
            out.setdefault(r.lead_id, [])
            if texts[r.issue_type] not in out[r.lead_id]:
                out[r.lead_id].append(texts[r.issue_type])
    return {k: "; ".join(v) for k, v in out.items()}


_RANK_FIRST_COLS = ["lead_id", "company", "stage", "deal_value", "days_overdue", "caveat"]


def rank_chase_first(flagged: pd.DataFrame, issues: pd.DataFrame | None = None,
                     limit: int = 5) -> pd.DataFrame:
    """Deterministic 'chase first' ranking: overdue, non-duplicate, open deals,
    by deal value then days overdue. Code (not the model) decides which rows appear."""
    df = flagged[flagged["is_overdue"] & ~flagged["is_duplicate"]].copy()
    df["deal_value"] = df["deal_value"].fillna(0)
    df = df.sort_values(["deal_value", "days_overdue", "lead_id"], ascending=[False, False, True])
    caveats = _caveats(issues) if issues is not None else {}
    df["caveat"] = df["lead_id"].map(caveats).fillna("")
    df = df.head(limit)
    cols = _RANK_FIRST_COLS + [c for c in df.columns if c not in _RANK_FIRST_COLS]
    return df[cols].reset_index(drop=True)


_ROW_COLS = ["lead_id", "company", "stage", "industry", "deal_value_bdt", "days_overdue",
             "next_follow_up", "caveat"]


def _rows_for_model(table: pd.DataFrame, limit: int = 15) -> list[dict]:
    rows = []
    for r in table.head(limit).to_dict("records"):
        r = {k: _s(v) for k, v in r.items()}
        r["deal_value_bdt"] = _bdt(r.get("deal_value"))
        rows.append({k: r.get(k) for k in _ROW_COLS if k in r or k == "deal_value_bdt"})
    return rows


class _Answer(dict):
    """{"text", "table", "filters"}. str() gives the text, so code that just does
    st.markdown(answer) keeps working unchanged."""

    def __str__(self) -> str:
        return self["text"]


def answer_question(question: str, df: pd.DataFrame) -> dict:
    flagged, issues, live = _frames(df)
    spec, err = _parse(question)
    empty = live.iloc[0:0]

    wants_rank = bool(_CHASE_RE.search(question))
    if wants_rank:  # code decides, whatever the model guessed (or if it was unreachable)
        spec = _empty_spec("rank") if err is not None else spec
        spec["intent"] = "rank"
        err = None
    if err is not None:
        return _Answer(text=FALLBACK_ANSWER + _hint(err), table=empty, filters=spec["filters"])
    if spec["intent"] == "unsupported":
        return _Answer(text=UNSUPPORTED_TEXT, table=empty, filters=spec["filters"])

    if spec["intent"] == "rank":
        table = rank_chase_first(flagged, issues, limit=spec["limit"] or 5)
        # show exactly what was applied, not the model's guess
        spec["filters"] = {**_empty_spec()["filters"], "overdue": True}
        spec["sort_by"] = "deal_value"
        n = len(table)
        if n == 0:
            return _Answer(text="No overdue deals right now.", table=table, filters=spec["filters"])
        header = (f"{n} overdue deal{'s' if n != 1 else ''} to chase first "
                  "(ranked by deal value, then days overdue):")
        facts = {"intent": "rank", "rows": _rows_for_model(table)}
        try:
            body = _call_llm(RANK_WRITER_SYSTEM_PROMPT,
                             "Ranked deals (JSON):\n" + json.dumps(facts, ensure_ascii=False),
                             temperature=0.2, max_tokens=1500)
        except AIUnavailable as e:
            body = "The table below has the details." + _hint(e)
        return _Answer(text=f"{header}\n\n{body}", table=table, filters=spec["filters"])

    table = apply_filters(live, issues, spec)
    total = float(table["deal_value"].fillna(0).sum())
    metrics = {"matched_leads": int(len(table)), "total_deal_value_bdt": _bdt(total)}
    facts = {"question": question, "intent": spec["intent"], "filters_applied": spec["filters"],
             "metrics": metrics, "rows": _rows_for_model(table),
             "rows_shown": min(len(table), 15), "rows_total": int(len(table))}
    try:
        text = _call_llm(ANSWER_WRITER_SYSTEM_PROMPT,
                         "Facts (JSON):\n" + json.dumps(facts, ensure_ascii=False),
                         temperature=0.2, max_tokens=1500)
    except AIUnavailable as e:
        text = (f"I couldn't reach the AI writer just now, but {metrics['matched_leads']} lead(s) match, "
                f"totalling {metrics['total_deal_value_bdt']}. The table has the details." + _hint(e))
    return _Answer(text=text, table=table, filters=spec["filters"])


# --------------------------------------------------------------------------
# 4. Weekly summary
# --------------------------------------------------------------------------
def data_problems_line(flagged: pd.DataFrame, issues: pd.DataFrame) -> str:
    """One line listing the data problems, written by code (the model never writes it)."""
    live_ids = set(flagged.loc[~flagged["is_duplicate"], "lead_id"])
    parts = []
    dups = sorted(flagged[flagged["is_duplicate"]].itertuples(), key=lambda r: engine._id_key(r.lead_id))
    if dups:
        inflation = _bdt(sum(0 if pd.isna(r.deal_value) else r.deal_value for r in dups))
        pairs = ", ".join(f"{r.lead_id} duplicates {r.duplicate_of}" for r in dups)
        parts.append(f"duplicates inflate the pipeline by {inflation} ({pairs})")
    for t, label in (("date_order", "date-order error"),
                     ("missing_contact", "missing contact details"),
                     ("invalid_email", "invalid email"),
                     ("stage_conflict", "stage conflict")):
        ids = sorted(set(issues.loc[issues["issue_type"] == t, "lead_id"]) & live_ids, key=engine._id_key)
        if ids:
            parts.append(f"{label}: {', '.join(ids)}")
    return ("Data problems to fix: " + "; ".join(parts) + ".") if parts else ""


def build_weekly_facts(df: pd.DataFrame) -> dict:
    flagged, issues, live = _frames(df)
    inflation = flagged.loc[flagged["is_duplicate"], "deal_value"].fillna(0).sum()
    note = live.set_index("lead_id")["notes"].map(lambda v: (_s(v) or "")[:140])

    def rows(frame, with_notes=False):
        out = []
        for r in frame.to_dict("records"):
            r = {k: _s(v) for k, v in r.items()}
            d = {"lead_id": r["lead_id"], "company": r["company"], "stage": r["stage"],
                 "deal_value_bdt": _bdt(r.get("deal_value")), "days_overdue": r.get("days_overdue"),
                 "next_follow_up": r.get("next_follow_up"), "caveat": r.get("caveat") or ""}
            if with_notes:
                d["notes"] = note.get(r["lead_id"])
            out.append(d)
        return out

    cold = live[live["is_going_cold"]].sort_values("days_since_contact", ascending=False)
    hard = issues[issues["issue_type"].isin(engine.HARD_TYPES) & issues["lead_id"].isin(live["lead_id"])]
    return {
        "chase_first": rows(rank_chase_first(flagged, issues), with_notes=True),
        "overdue_total_count": int(flagged["is_overdue"].sum()),
        "duplicate_inflation_bdt": _bdt(inflation),
        "going_cold": rows(cold),
        "hard_data_problems": [{"issue_type": t, "lead_ids": sorted(g["lead_id"].unique())}
                               for t, g in hard.groupby("issue_type")],
        "data_problems_line": data_problems_line(flagged, issues),
    }


def weekly_summary(df: pd.DataFrame) -> str:
    facts = build_weekly_facts(df)
    llm_facts = {k: v for k, v in facts.items() if k != "data_problems_line"}
    try:
        body = _call_llm(WEEKLY_SYSTEM_PROMPT,
                         "Facts (JSON):\n" + json.dumps(llm_facts, ensure_ascii=False),
                         temperature=0.3, max_tokens=2500)
    except AIUnavailable as e:
        return FALLBACK_SUMMARY + _hint(e)
    return f"{body}\n\n{facts['data_problems_line']}".strip()