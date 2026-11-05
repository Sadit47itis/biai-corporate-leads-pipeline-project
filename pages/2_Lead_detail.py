import streamlit as st

import ai
import engine
import ui

flagged, issues, clean = ui.get_data()
leads = flagged.sort_values("lead_id").reset_index(drop=True)
ids = leads["lead_id"].tolist()
labels = {r.lead_id: f"{r.lead_id} · {r.company}" for r in leads.itertuples()}

# Plain session-state containers survive page switches (widget keys do not).
st.session_state.setdefault("selected_lead", ids[0])
st.session_state.setdefault("drafts", {})
st.session_state.setdefault("reviewed", set())

ui.render_header()
st.title("Lead detail")


def _remember_selection():
    st.session_state["selected_lead"] = st.session_state["lead_picker"]


current = st.session_state["selected_lead"] if st.session_state["selected_lead"] in ids else ids[0]
st.selectbox("Pick a lead", ids, index=ids.index(current), key="lead_picker",
             format_func=labels.get, on_change=_remember_selection)
lead_id = st.session_state["lead_picker"]
st.session_state["selected_lead"] = lead_id

row = leads[leads["lead_id"] == lead_id].iloc[0]
mine = issues[issues["lead_id"] == lead_id]

# ---- header chips ----------------------------------------------------------
chips = [f'<span class="bdg grey">{ui.esc(row["stage"])}</span>']
if row["is_overdue"]:
    chips.append(f'<span class="bdg red">Overdue {int(row["days_overdue"])}d</span>')
if row["is_due_today"]:
    chips.append('<span class="bdg amber">Due today</span>')
if row["is_duplicate"]:
    chips.append(f'<span class="bdg grey">Duplicate of {ui.esc(row["duplicate_of"])}</span>')
if st.session_state["reviewed"] and lead_id in st.session_state["reviewed"]:
    chips.append('<span class="bdg green">Reviewed</span>')
st.markdown(f"### {ui.esc(row['company'])} " + " ".join(chips), unsafe_allow_html=True)


def _toggle_reviewed():
    if st.session_state[f"rev_{lead_id}"]:
        st.session_state["reviewed"].add(lead_id)
    else:
        st.session_state["reviewed"].discard(lead_id)


st.toggle("Mark as reviewed", value=lead_id in st.session_state["reviewed"],
          key=f"rev_{lead_id}", on_change=_toggle_reviewed)
st.caption(f"{len(st.session_state['reviewed'])} of {len(ids)} leads reviewed this session.")

# ---- all fields ------------------------------------------------------------
st.subheader("Details")
FIELDS = [
    ("Lead ID", "lead_id"), ("Company", "company"), ("Industry", "industry"),
    ("Contact name", "contact_name"), ("Contact role", "contact_role"),
    ("Contact email", "contact_email"), ("Contact phone", "contact_phone"),
    ("Source", "source"), ("Stage", "stage"), ("Deal value", "deal_value"),
    ("Last contact", "last_contact"), ("Next follow-up (original)", "next_follow_up"),
]


def kv(label, value, missing_ok=False):
    if ui.blank(value):
        shown = '<span class="kv-missing">missing</span>' if not missing_ok else "—"
    else:
        shown = ui.esc(value)
    return f'<div class="kv-l">{ui.esc(label)}</div><div class="kv-v">{shown}</div>'


with st.container(border=True):
    cols = st.columns(3)
    for i, (label, key) in enumerate(FIELDS):
        v = row[key]
        if key == "deal_value":
            v = ui.fmt_bdt(v)
        elif key in ("last_contact", "next_follow_up"):
            v = None if ui.blank(v) else ui.fmt_date(v)
        cols[i % 3].markdown(kv(label, v, missing_ok=key in ("last_contact", "next_follow_up")),
                             unsafe_allow_html=True)
    st.markdown(kv("Notes", row["notes"], missing_ok=True), unsafe_allow_html=True)

if not ui.blank(row["corrected_follow_up"]):
    st.markdown("**Follow-up date: original vs suggested**")
    a, b = st.columns(2)
    a.metric("Original (kept, untouched)", ui.fmt_date(row["next_follow_up"]))
    b.metric("Suggested corrected follow-up", ui.fmt_date(row["corrected_follow_up"]),
             help="TODAY + 3 days, pending confirmation. Never written over the original.")

# ---- issues ----------------------------------------------------------------
st.subheader("Issues and suggested fixes")
if mine.empty:
    st.success("No issues found for this lead.")
for r in mine.itertuples():
    tier = ui.issue_tier(r.issue_type)
    tag = "bdg orange" if tier == "Hard" else "bdg blue"
    st.markdown(
        f'<div class="issue {tier.lower()}">'
        f'<div class="issue-h">{ui.esc(ui.ISSUE_LABELS[r.issue_type])} '
        f'<span class="{tag}">{tier}</span> <span class="bdg grey">{ui.esc(r.severity)}</span></div>'
        f'<div class="issue-m">{ui.esc(r.message)}</div>'
        f'<div class="issue-f"><b>Suggested fix:</b> {ui.esc(r.suggested_fix)}</div></div>',
        unsafe_allow_html=True)

# ---- follow-up email -------------------------------------------------------
st.subheader("Follow-up email")
bad_email = mine[mine["issue_type"] == "invalid_email"]
if not bad_email.empty:
    st.error(f"**Fix email before sending.** {bad_email.iloc[0]['suggested_fix']}")

draft_key = f"draft_text_{lead_id}"


def _save_edit():
    st.session_state["drafts"][lead_id] = st.session_state[draft_key]


if st.button("Draft follow-up email", type="primary"):
    with st.spinner("Drafting..."):
        text = ai.draft_email(ui.lead_to_dict(row, issues))
    st.session_state["drafts"][lead_id] = text
    st.session_state[draft_key] = text

draft = st.session_state["drafts"].get(lead_id)
if draft is not None:
    if draft.startswith("HOLD:"):
        reason = ui.esc(draft[len("HOLD:"):].strip()).replace("\n", "<br>")
        st.markdown(f'<div class="hold"><b>On hold. Do not send yet.</b><br>{reason}</div>',
                    unsafe_allow_html=True)
    else:
        if "NOTE:" in draft:
            body, note = draft.split("NOTE:", 1)
            draft = body.strip()
            st.warning(f"**ACTION REQUIRED:** {note.strip()}")
        
        if draft_key not in st.session_state:
            st.session_state[draft_key] = draft
        st.text_area("Draft (editable)", key=draft_key, height=280, on_change=_save_edit)
else:
    st.caption("No draft yet. Click the button to generate one.")
