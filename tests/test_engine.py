import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine  # noqa: E402

TODAY = date(2026, 11, 1)


@pytest.fixture(scope="module")
def source():
    return engine.load_leads()


@pytest.fixture(scope="module")
def result(source):
    return engine.analyze(source, TODAY)


@pytest.fixture(scope="module")
def flagged(result):
    return result[0]


@pytest.fixture(scope="module")
def issues(result):
    return result[1]


def ids(issues, issue_type):
    return sorted(issues.loc[issues["issue_type"] == issue_type, "lead_id"])


def row(issues, lead_id, issue_type):
    r = issues[(issues["lead_id"] == lead_id) & (issues["issue_type"] == issue_type)]
    assert len(r) == 1, f"expected one {issue_type} row for {lead_id}, got {len(r)}"
    return r.iloc[0]


# ---------------------------------------------------------------- loading
def test_columns_are_snake_case(source):
    assert "lead_id" in source.columns
    assert "deal_value" in source.columns
    assert "next_follow_up" in source.columns
    assert pd.api.types.is_datetime64_any_dtype(source["next_follow_up"])
    assert len(source) == 25


# ---------------------------------------------------------------- overdue
def test_overdue_as_recorded_and_after_fix(flagged, issues):
    expected = ["L04", "L06", "L10", "L13", "L15", "L19", "L20", "L22", "L25"]
    assert ids(issues, "overdue") == expected
    counts = engine.overdue_counts(flagged)
    assert counts["as_recorded"] == 9
    # applying the L25 fix (follow-up -> TODAY+3) removes it from overdue
    assert counts["after_suggested_fixes"] == 8


def test_days_overdue_L15(flagged, issues):
    assert flagged.set_index("lead_id").loc["L15", "days_overdue"] == 13
    assert "13 days overdue" in row(issues, "L15", "overdue")["message"]


def test_due_today_is_not_overdue(flagged, issues):
    assert ids(issues, "due_today") == ["L09"]
    assert "L09" not in ids(issues, "overdue")
    assert not flagged.set_index("lead_id").loc["L09", "is_overdue"]


# ---------------------------------------------------------------- duplicates
def test_duplicates(flagged, issues):
    dup_rows = issues[issues["issue_type"] == "duplicate"]
    assert row(issues, "L03", "duplicate")["message"] == "Duplicate of L02"
    assert row(issues, "L03", "duplicate")["severity"] == "medium"
    assert row(issues, "L23", "duplicate")["message"] == "Duplicate of L01"
    assert row(issues, "L02", "duplicate")["message"] == "Surviving record; L03 is a duplicate."
    assert row(issues, "L02", "duplicate")["severity"] == "low"
    assert row(issues, "L01", "duplicate")["message"] == "Surviving record; L23 is a duplicate."
    assert sorted(dup_rows["lead_id"]) == ["L01", "L02", "L03", "L23"]
    assert "L17" not in set(dup_rows["lead_id"])
    flags = flagged.set_index("lead_id")["is_duplicate"]
    assert flags[["L03", "L23"]].all()
    assert not flags.drop(["L03", "L23"]).any()


def test_possible_duplicate_is_empty(issues):
    assert ids(issues, "possible_duplicate") == []


def test_blank_contacts_never_match_each_other():
    def lead(lid, company, email, phone):
        return {"lead id": lid, "company": company, "contact name": "X",
                "contact email": email, "contact phone": phone, "stage": "Contacted",
                "deal value (BDT)": 1000, "last contact": "2026-10-30",
                "next follow-up": "2026-11-05", "notes": ""}

    # both emails blank (blank string / None / whitespace), phones differ
    a = pd.DataFrame([lead("S1", "Alpha Textiles", "", "01711-000001"),
                      lead("S2", "Zenith Pharma", None, "01819-000002"),
                      lead("S3", "Quartz Logistics", "  ", "01911-000003")])
    # both phones blank, emails differ
    b = pd.DataFrame([lead("S1", "Alpha Textiles", "a@alpha.example", ""),
                      lead("S2", "Zenith Pharma", "z@zenith.example", None)])
    # everything blank
    c = pd.DataFrame([lead("S1", "Alpha Textiles", "", ""),
                      lead("S2", "Zenith Pharma", None, None)])
    for frame in (a, b, c):
        flagged, issues = engine.analyze(frame, TODAY)
        assert not flagged["is_duplicate"].any()
        assert ids(issues, "duplicate") == []

    # positive controls: a real shared email, or a real shared phone, still matches
    same_email = pd.DataFrame([lead("S1", "Alpha Textiles", "a@x.example", "01711-000001"),
                               lead("S2", "Zenith Pharma", "A@x.example", "01819-000002")])
    same_phone = pd.DataFrame([lead("S1", "Alpha Textiles", "a@x.example", "01711-000001"),
                               lead("S2", "Zenith Pharma", "z@y.example", "01711-000001")])
    for frame in (same_email, same_phone):
        _, issues = engine.analyze(frame, TODAY)
        assert row(issues, "S2", "duplicate")["message"] == "Duplicate of S1"


# ---------------------------------------------------------------- data quality
def test_date_order(flagged, issues):
    assert ids(issues, "date_order") == ["L25"]
    r = flagged.set_index("lead_id").loc["L25"]
    assert r["corrected_follow_up"] == pd.Timestamp("2026-11-04")  # TODAY + 3
    assert r["next_follow_up"] == pd.Timestamp("2026-10-21")       # original untouched
    assert "TODAY+3" in row(issues, "L25", "date_order")["suggested_fix"]


def test_missing_contact(issues):
    assert ids(issues, "missing_contact") == ["L06", "L24"]
    m06 = row(issues, "L06", "missing_contact")["message"]
    assert "contact_phone" in m06 and "contact_email" not in m06 and "contact_name" not in m06
    m24 = row(issues, "L24", "missing_contact")["message"]
    assert "contact_name" in m24 and "contact_email" in m24 and "contact_phone" not in m24


def test_invalid_email(issues):
    assert ids(issues, "invalid_email") == ["L18"]
    assert "shahidul@boguratextile.example" in row(issues, "L18", "invalid_email")["suggested_fix"]


def test_stage_conflict(issues):
    assert ids(issues, "stage_conflict") == ["L12", "L18"]
    fix = row(issues, "L12", "stage_conflict")["suggested_fix"]
    assert fix == "Likely Invoice sent for the balance. Not applied, needs a human."


# ---------------------------------------------------------------- watch list
def test_payment_check(issues):
    assert ids(issues, "payment_check") == ["L07", "L22"]


def test_going_cold(issues):
    assert ids(issues, "going_cold") == ["L06", "L15", "L22"]
    assert "L12" not in ids(issues, "going_cold")  # Paid is an end stage


# ---------------------------------------------------------------- reporting
def test_clean_leads(source):
    clean = engine.clean_leads(source)
    assert len(clean) == 23
    assert {"L01", "L02"} <= set(clean["lead_id"])
    assert not {"L03", "L23"} & set(clean["lead_id"])
    assert len(source) == 25  # originals preserved


def test_stage_totals_after_dedup(source):
    summary = engine.stage_summary(engine.clean_leads(source)).set_index("stage")
    totals = summary["total_deal_value"]
    assert totals["Proposal"] == 920000
    assert totals["Meeting"] == 1550000
    assert totals["Invoice sent"] == 400000
    assert totals["Paid"] == 300000


def test_scorecard(source, issues):
    card = engine.scorecard(issues, engine.clean_leads(source))
    assert card["leads_with_hard_issues"] == 16
    assert card["unique_companies"] == 23


# ---------------------------------------------------------------- invariants
def test_source_is_never_mutated():
    src = engine.load_leads()
    before = src.copy(deep=True)
    engine.analyze(src, TODAY)
    engine.clean_leads(src)
    pd.testing.assert_frame_equal(src, before)


def test_today_is_a_parameter(source):
    # On 2026-10-24: L22's follow-up (10-24) is due today; L06 (10-22), L15 (10-19)
    # and L25 (10-21) are strictly before it.
    _, issues = engine.analyze(source, date(2026, 10, 24))
    assert ids(issues, "due_today") == ["L22"]
    assert ids(issues, "overdue") == ["L06", "L15", "L25"]