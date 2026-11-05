"""Offline tests for ai.py: a fake client stands in for Groq, so no key or network is needed."""
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai  # noqa: E402
import engine  # noqa: E402

CHASE_FIRST = ["L10", "L22", "L19", "L15", "L25"]   # overdue, by deal value then days overdue
OK_EMAIL = "test@example.com"


class FakeClient:
    """Pops scripted replies (str -> reply text, Exception -> raised) and records every call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


@pytest.fixture
def fake(monkeypatch):
    sleeps = []
    monkeypatch.setattr(ai.time, "sleep", lambda s: sleeps.append(s))

    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(ai, "_get_client", lambda: client)
        client.sleeps = sleeps
        return client
    return install


@pytest.fixture(scope="module")
def flagged():
    return engine.analyze(engine.load_leads(), date(2026, 11, 1))[0]


def spec(intent="list", sort_by=None, limit=None, **filters):
    base = dict(stage=None, overdue=None, industry=None, has_issue=None,
                missing_contact=None, min_value=None)
    base.update(filters)
    return json.dumps({"intent": intent, "filters": base, "sort_by": sort_by, "limit": limit})


def ask(fake, flagged, parsed, question="q"):
    client = fake([parsed, "Answer text."])
    return ai.answer_question(question, flagged), client


def ids(result):
    return list(result["table"]["lead_id"])


# ---------------------------------------------------------------- email
def test_email_uses_verbatim_prompt_and_only_listed_fields(fake):
    client = fake(["Subject: Hi\n\nBody"])
    lead = {"company": "Meghna Pharma Ltd", "contact_name": "Dr. Sultana Begum", "contact_role": "Head",
            "stage": "Proposal", "deal_value": 700000, "last_contact": "2026-10-27",
            "next_follow_up": "2026-11-04", "notes": "Proposal sent.",
            "contact_email": "secret@x.example", "contact_phone": "01819-200102",
            "issues": [{"issue_type": "duplicate"}, {"issue_type": "going_cold"}]}
    assert ai.draft_email(lead).startswith("Subject:")
    call = client.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["messages"][0] == {"role": "system", "content": ai.EMAIL_SYSTEM_PROMPT}
    for phrase in ("70-120 words", "signed 'BIAI Team'", "HOLD rule", "starting with 'NOTE:'", "No markdown.",
                   "Never state dates"):
        assert phrase in ai.EMAIL_SYSTEM_PROMPT
    user = call["messages"][1]["content"]
    assert "secret@x.example" not in user and "01819-200102" not in user
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["hard_issue_types"] == ["duplicate"]  # watch types are not "hard"
    assert payload["notes"] == "Proposal sent."


def test_lost_and_duplicate_leads_are_held_without_calling_the_model(fake):
    client = fake([])
    assert ai.draft_email({"stage": "Lost", "notes": "Do not contact again"}).startswith("HOLD:")
    assert ai.draft_email({"stage": "Meeting", "is_duplicate": True,
                           "duplicate_of": "L02"}).startswith("HOLD:")
    assert client.calls == []


def test_missing_email_is_held_with_phone_and_no_model_call(fake):
    client = fake([])
    out = ai.draft_email({"stage": "New", "contact_phone": "01555-200124"})
    assert out == "HOLD: No email address on file. Call 01555-200124 instead."
    assert ai.draft_email({"stage": "New", "contact_email": float("nan")}).startswith("HOLD: No email")
    assert ai.draft_email({"stage": "New", "contact_email": "  "}).startswith("HOLD: No email")
    assert client.calls == []


def test_check_bank_notes_are_held_in_code_not_left_to_the_model(fake):
    client = fake([])
    out = ai.draft_email({"stage": "Invoice sent", "contact_email": OK_EMAIL,
                          "notes": "Invoice sent 17 Oct. Paid? Not marked. Check bank before chasing."})
    assert out.startswith("HOLD:") and client.calls == []
    # L07-style notes ask a question instead, so they still get a draft
    client = fake(["Subject: ok"])
    assert ai.draft_email({"stage": "Invoice sent", "contact_email": OK_EMAIL,
                           "notes": "Invoice sent, waiting for payment. Ask whether the advance has come in."
                           }) == "Subject: ok"


def test_retry_once_with_two_second_backoff(fake):
    client = fake([RuntimeError("boom"), "Subject: ok"])
    assert ai.draft_email({"stage": "Meeting", "contact_email": OK_EMAIL}) == "Subject: ok"
    assert len(client.calls) == 2 and client.sleeps == [2]


def test_retry_goes_to_the_fallback_model(fake):
    client = fake([RuntimeError("429"), "Subject: ok"])
    ai.draft_email({"stage": "Meeting", "contact_email": OK_EMAIL})
    assert [c["model"] for c in client.calls] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def test_two_failures_return_friendly_fallback(fake):
    client = fake([RuntimeError("a"), RuntimeError("b")])
    out = ai.draft_email({"stage": "Meeting", "contact_email": OK_EMAIL})
    assert out.startswith("[Draft unavailable]") and not out.startswith("HOLD:")
    assert len(client.calls) == 2 and client.sleeps == [2]


def test_missing_key_falls_back_without_retrying(monkeypatch):
    monkeypatch.setattr(ai, "_api_key", lambda: None)
    monkeypatch.setattr(ai, "_client", None)
    monkeypatch.setattr(ai.time, "sleep", lambda s: pytest.fail("must not back off"))
    out = ai.draft_email({"stage": "Meeting", "contact_email": OK_EMAIL})
    assert "[Draft unavailable]" in out and "GROQ_API_KEY" in out


# ---------------------------------------------------------------- parser
def test_parse_question_requests_json_and_normalises(fake):
    raw = json.dumps({"intent": "RANK?", "filters": {"stage": ["invoice_sent", "Nonsense"], "overdue": "yes",
                                                     "min_value": True},
                      "sort_by": "colour", "limit": -3})
    client = fake([raw])
    out = ai.parse_question("anything")
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["messages"][0]["content"] == ai.QUESTION_PARSER_SYSTEM_PROMPT
    assert out["intent"] == "unsupported"
    assert out["filters"]["stage"] == ["Invoice sent"]
    assert out["filters"]["overdue"] is None and out["filters"]["min_value"] is None
    assert out["sort_by"] is None and out["limit"] is None


def test_parse_question_survives_garbage_and_failures(fake):
    fake(["not json at all"])
    assert ai.parse_question("x")["intent"] == "unsupported"
    fake([RuntimeError("a"), RuntimeError("b")])
    assert "error" in ai.parse_question("x")


# ---------------------------------------------------------------- answer_question
def test_answer_sum_is_computed_in_pandas_and_dedupes(fake, flagged):
    res, client = ask(fake, flagged, spec("sum", stage=["Proposal"]))
    assert ids(res) == ["L02", "L14"]                  # L03 (duplicate) excluded
    assert res["table"]["deal_value"].sum() == 920000
    assert set(res) == {"text", "table", "filters"} and str(res) == "Answer text."
    assert res["filters"]["stage"] == ["Proposal"]
    writer = client.calls[1]
    assert writer["messages"][0]["content"] == ai.ANSWER_WRITER_SYSTEM_PROMPT
    assert "BDT 920,000" in writer["messages"][1]["content"]
    assert "response_format" not in writer


# Spec change (deliberate): "chase first" = OVERDUE, non-duplicate deals ranked by deal value,
# then days overdue. The old test ranked all open deals by days overdue, which put four BDT 0
# deals ahead of the BDT 500,000 deal and disagreed with the weekly summary.
def test_answer_rank_chase_first(fake, flagged):
    res, client = ask(fake, flagged, spec("rank", sort_by="days_overdue"))
    assert ids(res) == CHASE_FIRST
    assert str(res).startswith("5 overdue deals to chase first")
    assert res["filters"]["overdue"] is True and res["filters"]["stage"] is None
    caveats = res["table"].set_index("lead_id")["caveat"]
    assert "verify payment" in caveats["L22"] and "date suspect" in caveats["L25"] and caveats["L10"] == ""
    assert client.calls[1]["messages"][0]["content"] == ai.RANK_WRITER_SYSTEM_PROMPT
    assert "verify payment" in client.calls[1]["messages"][1]["content"]   # caveat reaches the writer


def test_rank_rows_are_chosen_by_code_not_by_the_parser(fake, flagged):
    # whatever stages or sort the model guessed, the rows are the same
    res, _ = ask(fake, flagged, spec("rank", stage=["Meeting", "Proposal"], sort_by="deal_value"))
    assert ids(res) == CHASE_FIRST


def test_chase_wording_forces_rank_even_if_parser_says_list_or_unsupported(fake, flagged):
    for parsed in (spec("list", stage=["Meeting"]), spec("unsupported")):
        res, _ = ask(fake, flagged, parsed, question="Which deals should I chase first?")
        assert ids(res) == CHASE_FIRST


def test_chase_first_still_works_when_the_parser_is_down(fake, flagged):
    fake([RuntimeError("a"), RuntimeError("b"), "L10 ..."])
    res = ai.answer_question("Which deals should I chase first?", flagged)
    assert ids(res) == CHASE_FIRST and str(res).startswith("5 overdue deals")


def test_chase_first_when_the_writer_is_down_keeps_the_header_and_table(fake, flagged):
    fake([spec("rank"), RuntimeError("a"), RuntimeError("b")])
    res = ai.answer_question("q", flagged)
    assert str(res).startswith("5 overdue deals to chase first") and ids(res) == CHASE_FIRST


def test_answer_overdue_missing_industry_and_value_filters(fake, flagged):
    assert sorted(ids(ask(fake, flagged, spec(overdue=True))[0])) == \
        ["L04", "L06", "L10", "L13", "L15", "L19", "L20", "L22", "L25"]
    assert sorted(ids(ask(fake, flagged, spec(missing_contact=True))[0])) == ["L06", "L24"]
    assert sorted(ids(ask(fake, flagged, spec(industry=["pharma"]))[0])) == ["L02", "L20"]
    assert ids(ask(fake, flagged, spec(sort_by="deal_value", min_value=400000))[0]) == ["L02", "L10", "L17"]
    has_issue = set(ids(ask(fake, flagged, spec(has_issue=True))[0]))
    assert len(has_issue) == 14 and not {"L03", "L23"} & has_issue   # 16 minus the two removed duplicates


def test_answer_unsupported_makes_one_call(fake, flagged):
    client = fake([spec("unsupported")])
    res = ai.answer_question("What's the weather?", flagged)
    assert res["text"] == ai.UNSUPPORTED_TEXT and res["table"].empty and len(client.calls) == 1


def test_answer_when_writer_fails_still_states_computed_numbers(fake, flagged):
    fake([spec("sum", stage=["Proposal"]), RuntimeError("a"), RuntimeError("b")])
    res = ai.answer_question("q", flagged)
    assert "BDT 920,000" in res["text"] and len(res["table"]) == 2


def test_answer_when_parser_fails_returns_fallback(fake, flagged):
    fake([RuntimeError("a"), RuntimeError("b")])
    res = ai.answer_question("q", flagged)
    assert res["text"].startswith("Sorry, I couldn't reach") and res["table"].empty


# ---------------------------------------------------------------- weekly summary
def test_weekly_facts_are_computed_not_guessed(flagged):
    f = ai.build_weekly_facts(flagged)
    assert [d["lead_id"] for d in f["chase_first"]] == CHASE_FIRST   # same ranking as the Ask screen
    assert f["chase_first"][0]["deal_value_bdt"] == "BDT 500,000"
    caveats = {d["lead_id"]: d["caveat"] for d in f["chase_first"]}
    assert "verify payment" in caveats["L22"] and "date suspect" in caveats["L25"] and caveats["L10"] == ""
    assert f["overdue_total_count"] == 9
    assert f["duplicate_inflation_bdt"] == "BDT 1,050,000"          # L03 (700k) + L23 (350k)
    assert [d["lead_id"] for d in f["going_cold"]] == ["L15", "L06", "L22"]
    types = {p["issue_type"]: p["lead_ids"] for p in f["hard_data_problems"]}
    assert types["date_order"] == ["L25"] and types["invalid_email"] == ["L18"]
    assert "L03" not in types["duplicate"] and "L23" not in types["duplicate"]


def test_weekly_summary_appends_a_data_problems_line_written_by_code(fake, flagged):
    client = fake(["1. L10 ..."])
    out = ai.weekly_summary(flagged)
    assert out.startswith("1. L10 ...")
    assert "BDT 1,050,000" in out and "L03 duplicates L02" in out and "L23 duplicates L01" in out
    assert "date-order error: L25" in out and "invalid email: L18" in out
    assert "missing contact details: L06, L24" in out
    system = client.calls[0]["messages"][0]["content"]
    assert "150 words" in system and "unpaid" in system
    assert "data_problems_line" not in client.calls[0]["messages"][1]["content"]   # model never sees it


def test_weekly_summary_falls_back(fake, flagged):
    fake([RuntimeError("a"), RuntimeError("b")])
    assert "couldn't be generated" in ai.weekly_summary(flagged)