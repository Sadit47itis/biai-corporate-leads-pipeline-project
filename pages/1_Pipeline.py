import pandas as pd
import streamlit as st

import engine
import ui

flagged, issues, clean = ui.get_data()

st.title("Pipeline")
st.markdown(f'<div class="scorecard">{ui.esc(ui.scorecard_line(flagged, issues, clean))}</div>',
            unsafe_allow_html=True)

# ---- metrics (whole pipeline, not affected by the board filters) -----------
oc = engine.overdue_counts(flagged)
sc = engine.scorecard(issues, clean)
open_value = clean.loc[clean["stage"].isin(engine.STAGES), "deal_value"].sum()
n_dups = int(flagged["is_duplicate"].sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Open pipeline value", ui.fmt_bdt(open_value))
m2.metric("Overdue (as recorded)", oc["as_recorded"])
m3.metric("Overdue (after fixes)", oc["after_suggested_fixes"])
m4.metric("Leads with data issues", sc["leads_with_hard_issues"])
m5.metric("Duplicates removed", n_dups)

# ---- filters (board only) --------------------------------------------------
st.write("")
f1, f2 = st.columns([2, 1])
industries = sorted(flagged["industry"].dropna().unique())
chosen = f1.multiselect("Industry", industries, placeholder="All industries")
search = f2.text_input("Search company", placeholder="e.g. Padma")

view = flagged
if chosen:
    view = view[view["industry"].isin(chosen)]
if search.strip():
    view = view[view["company"].str.contains(search.strip(), case=False, na=False, regex=False)]

st.markdown(
    '<div class="legend">'
    '<span class="bdg red">Overdue Nd</span> '
    '<span class="bdg amber">Due today</span> '
    '<span class="bdg grey">Duplicate</span> '
    '<span class="bdg orange">⚠ hard data issue</span> '
    '<span class="dot"></span> watch item (going cold / payment check). '
    'Column totals exclude duplicates. Filters apply to the board only.'
    '</div>', unsafe_allow_html=True)

# ---- per-lead badge inputs -------------------------------------------------
hard_other = {}   # hard issues that do not already have their own badge
watch = {}
for r in issues.itertuples():
    if r.issue_type in engine.HARD_TYPES and r.issue_type not in ("overdue", "duplicate"):
        hard_other.setdefault(r.lead_id, []).append(ui.ISSUE_LABELS[r.issue_type])
    elif r.issue_type in engine.WATCH_TYPES and r.issue_type != "due_today":
        watch.setdefault(r.lead_id, []).append(ui.ISSUE_LABELS[r.issue_type])


def card_html(r) -> str:
    badges = []
    if r.is_overdue:
        badges.append(f'<span class="bdg red">Overdue {int(r.days_overdue)}d</span>')
    if r.is_due_today:
        badges.append('<span class="bdg amber">Due today</span>')
    if r.is_duplicate:
        badges.append(f'<span class="bdg grey" title="Duplicate of {ui.esc(r.duplicate_of)}">Duplicate</span>')
    if r.lead_id in hard_other:
        tip = ui.esc(", ".join(hard_other[r.lead_id]))
        n = len(hard_other[r.lead_id])
        badges.append(f'<span class="bdg orange" title="{tip}">⚠ {ui.plural(n, "issue")}</span>')
    if r.lead_id in watch:
        badges.append(f'<span class="dot" title="{ui.esc(", ".join(watch[r.lead_id]))}"></span>')
    contact = r.contact_name if not ui.blank(r.contact_name) else "No contact name"
    industry = r.industry if not ui.blank(r.industry) else "—"
    return (
        f'<div class="lc{" dup" if r.is_duplicate else ""}">'
        f'<div class="lc-co">{ui.esc(r.company)}<span class="lc-id">{ui.esc(r.lead_id)}</span></div>'
        f'<div class="lc-sub">{ui.esc(contact)} · {ui.esc(industry)}</div>'
        f'<div class="lc-val">{ui.esc(ui.fmt_bdt(r.deal_value))}</div>'
        f'<div class="lc-fu">Next follow-up: {ui.esc(ui.fmt_date(r.next_follow_up))}</div>'
        f'<div class="bdgs">{"".join(badges)}</div>'
        '</div>'
    )


# ---- kanban ----------------------------------------------------------------
cols = st.columns(len(engine.ALL_STAGES))
for col, stage in zip(cols, engine.ALL_STAGES):
    items = view[view["stage"] == stage].sort_values(
        ["next_follow_up", "lead_id"], na_position="last")
    counted = items[~items["is_duplicate"]]
    n_dup_here = int(items["is_duplicate"].sum())
    note = f'<div class="kh-note">+{n_dup_here} duplicate (not counted)</div>' if n_dup_here else ""
    with col:
        st.markdown(
            f'<div class="kh" style="border-top-color:{ui.STAGE_COLORS[stage]}">'
            f'<div class="kh-top"><span class="kh-name">{stage}</span>'
            f'<span class="kh-count">{len(counted)}</span></div>'
            f'<div class="kh-val">{ui.esc(ui.fmt_bdt(counted["deal_value"].sum()))}</div>{note}</div>',
            unsafe_allow_html=True)
        if items.empty:
            st.caption("No leads")
        else:
            st.markdown("".join(card_html(r) for r in items.itertuples()), unsafe_allow_html=True)
