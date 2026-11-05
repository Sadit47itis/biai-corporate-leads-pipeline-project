"""AI layer: OpenAI Python SDK pointed at Groq (model: openai/gpt-oss-120b).

Division of labour: the model only (1) writes emails, (2) turns a question
into a JSON filter spec, (3) words an answer from numbers we computed, and
(4) writes the weekly "chase first" brief from facts we computed. Every number
comes from pandas / engine.py, never from the model.

Key lookup: st.secrets["GROQ_API_KEY"] if available, else the environment /
.env file (python-dotenv). Calls are wrapped with a 2-second backoff and one
retry; on failure each function returns a friendly fallback instead of raising.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

import engine

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"
BACKOFF_SECONDS = 2
MAX_ATTEMPTS = 2          # first try + 1 retry
TIMEOUT_SECONDS = 45

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
EMAIL_SYSTEM_PROMPT = """You write short follow-up emails for BIAI, a Bangladeshi AI training company, to corporate leads. Polite, warm, professional English, 70-120 words. Structure: greeting with the contact's name; one line referring to the specific last interaction in the notes; one clear next step with a suggested action (not a vague 'let me know'); polite close signed 'BIAI Team'. Rules: use ONLY facts in the lead data and notes; never invent prices, dates or promises. Stage Invoice sent: be gentle, ask whether they need anything from us to complete payment, never demand it. Stage Proposal: offer a quick call to walk through it. Stage Meeting: offer next steps toward a proposal. Stage Paid: a light note is fine, reference training kickoff, never request payment. HOLD rule: if the notes say the lead is lost or 'do not contact', or say to check internal records (for example 'check bank') before chasing, do NOT write an email; output one line starting with 'HOLD:' explaining why. If the contact name is missing, use 'Hello' and add a final line starting with 'NOTE:' saying the contact name and email must be confirmed. Output 'Subject: ...' first, then the body. No markdown."""

QUESTION_PARSER_SYSTEM_PROMPT = """You convert a user's question about a sales pipeline into the JSON schema provided. Stages: New, Contacted, Meeting, Proposal, Invoice sent, Paid, Lost. 'Overdue' means the follow-up date is before the reference date. 'Stuck' or 'chase first' means rank open deals by days_overdue then deal_value. The user may write in English, Bangla or Banglish. Do not answer the question. Output JSON only."""

ANSWER_WRITER_SYSTEM_PROMPT = """You are a pipeline assistant. Answer in 1-2 plain sentences using ONLY the numbers and rows provided. Currency is BDT, formatted with commas. Never compute or guess numbers yourself."""

# Not specified in the brief; written to follow the same rules (facts only, no maths).
WEEKLY_SYSTEM_PROMPT = (
    "You write the weekly pipeline brief for BIAI, a Bangladeshi AI training company. "
    "Using ONLY the facts provided, write a short ranked 'chase first' list of at most 150 words in total. "
    "Number the items. Each item: lead ID and company, one short reason, one concrete action. "
    "Currency is BDT with commas. Never compute or guess numbers; quote them as given. "
    "If a lead's notes say to check internal records (for example the bank) or not to contact, "
    "say to verify first instead of chasing. End with one line on the most important data problem to fix. "
    "No preamble."
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
    """One chat completion with a 2 s backoff and a single retry. Raises AIUnavailable."""
    client = _get_client()
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
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
    # Cheap deterministic guards (no model call): never draft for these.
    if lead.get("stage") == "Lost":
        return ("HOLD: This lead is marked Lost, so no follow-up should go out. "
                f"Notes say: {lead.get('notes') or 'no notes'}")
    if lead.get("is_duplicate"):
        return (f"HOLD: This record is a duplicate of {lead.get('duplicate_of')}. "
                "Email the surviving record instead so the contact is not messaged twice.")

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
    elif spec["intent"] == "rank":  # "chase first" ranks OPEN deals
        out = out[~out["stage"].isin(engine.END_STAGES)]
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

    sort_by = spec["sort_by"] or ("days_overdue" if spec["intent"] == "rank" else None)
    if sort_by == "days_overdue":
        out = out.sort_values(["days_overdue", "deal_value", "lead_id"], ascending=[False, False, True])
    elif sort_by == "deal_value":
        out = out.sort_values(["deal_value", "days_overdue", "lead_id"], ascending=[False, False, True])
    elif sort_by == "next_follow_up":
        out = out.sort_values(["next_follow_up", "lead_id"], na_position="last")
    limit = spec["limit"] or (5 if spec["intent"] == "rank" else None)
    if limit:
        out = out.head(limit)
    return out.reset_index(drop=True)


_ROW_COLS = ["lead_id", "company", "stage", "industry", "deal_value_bdt", "days_overdue", "next_follow_up"]


def _rows_for_model(table: pd.DataFrame, limit: int = 15) -> list[dict]:
    rows = []
    for r in table.head(limit).to_dict("records"):
        r = {k: _s(v) for k, v in r.items()}
        r["deal_value_bdt"] = _bdt(r.get("deal_value"))
        rows.append({k: r.get(k) for k in _ROW_COLS})
    return rows


class _Answer(dict):
    """{"text", "table", "filters"}. str() gives the text, so code that just does
    st.markdown(answer) keeps working unchanged."""

    def __str__(self) -> str:
        return self["text"]


def answer_question(question: str, df: pd.DataFrame) -> dict:
    _, issues, live = _frames(df)
    spec, err = _parse(question)
    empty = live.iloc[0:0]

    if err is not None:
        return _Answer(text=FALLBACK_ANSWER + _hint(err), table=empty, filters=spec["filters"])
    if spec["intent"] == "unsupported":
        return _Answer(text=UNSUPPORTED_TEXT, table=empty, filters=spec["filters"])

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
        # The numbers are already computed, so state them plainly instead of failing.
        text = (f"I couldn't reach the AI writer just now, but {metrics['matched_leads']} lead(s) match, "
                f"totalling {metrics['total_deal_value_bdt']}. The table has the details." + _hint(e))
    return _Answer(text=text, table=table, filters=spec["filters"])


# --------------------------------------------------------------------------
# 4. Weekly summary
# --------------------------------------------------------------------------
def build_weekly_facts(df: pd.DataFrame) -> dict:
    """Facts for the weekly brief, all computed in pandas on de-duplicated rows."""
    _, issues, live = _frames(df)
    note = live.set_index("lead_id")["notes"].map(lambda v: (_s(v) or "")[:140])

    def rows(frame, extra=(), with_notes=False):
        out = []
        for r in frame.to_dict("records"):
            r = {k: _s(v) for k, v in r.items()}
            d = {"lead_id": r["lead_id"], "company": r["company"], "stage": r["stage"],
                 "deal_value_bdt": _bdt(r.get("deal_value")), "next_follow_up": r.get("next_follow_up")}
            d.update({k: r.get(k) for k in extra})
            if with_notes:
                d["notes"] = note.get(r["lead_id"])
            out.append(d)
        return out

    overdue = live[live["is_overdue"]].sort_values(["deal_value", "days_overdue"], ascending=False).head(5)
    pipeline = live[live["stage"].isin(["Proposal", "Invoice sent"])]
    cold = live[live["is_going_cold"]].sort_values("days_since_contact", ascending=False)
    hard = issues[issues["issue_type"].isin(engine.HARD_TYPES) & issues["lead_id"].isin(live["lead_id"])]
    return {
        "top_overdue_by_value": rows(overdue, ("days_overdue",), with_notes=True),
        "overdue_total_count": int(live["is_overdue"].sum()),
        "proposal_and_invoice_sent": rows(pipeline, with_notes=True),
        "going_cold": rows(cold, ("days_since_contact",)),
        "hard_data_problems": [{"issue_type": t, "lead_ids": sorted(g["lead_id"].unique())}
                               for t, g in hard.groupby("issue_type")],
    }


def weekly_summary(df: pd.DataFrame) -> str:
    facts = build_weekly_facts(df)
    try:
        return _call_llm(WEEKLY_SYSTEM_PROMPT,
                         "Facts (JSON):\n" + json.dumps(facts, ensure_ascii=False),
                         temperature=0.3, max_tokens=2500)
    except AIUnavailable as e:
        return FALLBACK_SUMMARY + _hint(e)