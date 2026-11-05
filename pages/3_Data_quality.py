import pandas as pd
import streamlit as st

import engine
import ui

flagged, issues, clean = ui.get_data()
today = pd.Timestamp(ui.get_today())
company = flagged.set_index("lead_id")["company"]

st.session_state.setdefault("apply_fixes", False)

head, dl = st.columns([4, 1])
head.title("Data quality")
export = issues.assign(tier=issues["issue_type"].map(ui.issue_tier),
                       company=issues["lead_id"].map(company))
export = export[["lead_id", "company", "tier", "issue_type", "severity", "message", "suggested_fix"]]
dl.write("")
dl.download_button("⬇ Download issues (CSV)", export.to_csv(index=False).encode("utf-8"),
                   file_name="lead_issues.csv", mime="text/csv", width="stretch")

hard = issues[issues["issue_type"].isin(engine.HARD_TYPES)]
watch = issues[issues["issue_type"].isin(engine.WATCH_TYPES)]
c1, c2, c3 = st.columns(3)
c1.metric("Hard issues (counted in scorecard)", len(hard), help="Issue rows across the 6 hard types.")
c2.metric("Leads with a hard issue", hard["lead_id"].nunique())
c3.metric("Watch items (not counted)", len(watch))
st.caption(f"{len(hard)} issues across {hard['lead_id'].nunique()} leads")

# ---- how duplicates and the date error were handled --------------------------
with st.expander("How I handled duplicates and the date error", expanded=True):
    st.markdown("**Duplicates.** Leads that share the same non-blank email or phone are grouped. "
                "The lowest (earliest) lead ID is kept as the surviving record; the others are "
                "excluded from totals. Nothing is deleted: the source rows are preserved and the "
                "duplicate stays visible on the board, greyed out.")
    dup_rows = flagged[flagged["is_duplicate"]].sort_values("lead_id")
    for r in dup_rows.itertuples():
        keep = flagged[flagged["lead_id"] == r.duplicate_of].iloc[0]
        same = [n for n, k in (("email", "contact_email"), ("phone", "contact_phone"))
                if not ui.blank(getattr(r, k)) and str(getattr(r, k)).lower() == str(keep[k]).lower()]
        name_note = ("same company name" if r.company == keep["company"]
                     else f"company written differently (“{r.company}” vs “{keep['company']}”)")
        st.markdown(f"- **{r.lead_id} → {r.duplicate_of}**: same {' and '.join(same)}, {name_note}. "
                    f"{r.lead_id} is excluded from totals.")
    fixed = flagged[flagged["corrected_follow_up"].notna()].sort_values("lead_id")
    for r in fixed.itertuples():
        st.markdown(
            f"**{r.lead_id} date error.** Next follow-up ({ui.fmt_date(r.next_follow_up)}) is before "
            f"last contact ({ui.fmt_date(r.last_contact)}). I kept the last contact because it is a "
            f"recorded event, and suggest **{ui.fmt_date(r.corrected_follow_up)}** (today + 3) for the "
            "follow-up, pending confirmation. The original date is untouched in the data.")

# ---- apply suggested fixes -------------------------------------------------
st.subheader("Overdue: as recorded vs after fixes")


def _sync_fixes():
    st.session_state["apply_fixes"] = st.session_state["apply_fixes_toggle"]


apply = st.toggle("Apply suggested fixes", value=st.session_state["apply_fixes"],
                  key="apply_fixes_toggle", on_change=_sync_fixes,
                  help="Recomputes overdue using the corrected follow-up dates. Display only.")
st.session_state["apply_fixes"] = apply

follow = flagged["corrected_follow_up"].fillna(flagged["next_follow_up"]) if apply \
    else flagged["next_follow_up"]
is_end = flagged["stage"].isin(engine.END_STAGES)
mask = (follow < today) & ~is_end
as_recorded = int(flagged["is_overdue"].sum())
n = int(mask.sum())
m1, m2 = st.columns([1, 3])
m1.metric("Overdue leads", n, delta=(n - as_recorded) if apply and n != as_recorded else None,
          delta_color="inverse", help="Due exactly today is 'due today', not overdue.")
shown = flagged[mask].assign(follow_up_used=follow[mask],
                             days_overdue_shown=(today - follow[mask]).dt.days)
if shown.empty:
    m2.info("Nothing overdue.")
else:
    table = pd.DataFrame({
        "Lead": shown["lead_id"], "Company": shown["company"], "Stage": shown["stage"],
        "Follow-up used": shown["follow_up_used"].dt.strftime("%Y-%m-%d"),
        "Days overdue": shown["days_overdue_shown"].astype(int),
    }).sort_values("Days overdue", ascending=False)
    m2.dataframe(table, hide_index=True, width="stretch")
if apply:
    moved = flagged[flagged["is_overdue"] & ~mask]
    for r in moved.itertuples():
        st.caption(f"{r.lead_id} left the overdue list: follow-up moved to "
                   f"{ui.fmt_date(r.corrected_follow_up)}.")


# ---- grouped issues --------------------------------------------------------
def render_group(title, types, frame, caption):
    st.subheader(title)
    st.caption(caption)
    for t in types:
        rows = frame[frame["issue_type"] == t]
        label = f"{ui.ISSUE_LABELS[t]} · {len(rows)}"
        with st.expander(label, expanded=False):
            if rows.empty:
                st.write("None.")
                continue
            if t == "duplicate":
                st.caption(f"{int((rows['severity'] == 'medium').sum())} duplicate(s) and "
                           f"{int((rows['severity'] == 'low').sum())} surviving record(s).")
            st.dataframe(
                rows.assign(company=rows["lead_id"].map(company))[
                    ["lead_id", "company", "severity", "message", "suggested_fix"]],
                hide_index=True, width="stretch")


render_group("Hard issues", engine.HARD_TYPES, hard, "Counted in the scorecard.")
render_group("Watch items", engine.WATCH_TYPES, watch,
             "Shown in the UI for attention, not counted in the scorecard.")
