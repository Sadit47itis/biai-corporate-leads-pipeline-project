import streamlit as st

import ai
import ui

flagged, issues, clean = ui.get_data()

st.session_state.setdefault("chat", [])
st.session_state.setdefault("pending_q", None)

ui.render_header()

st.title("Ask your pipeline")
st.caption("Ask a question about the leads. The AI uses the rules engine for the math, "
           "and writes the answer in plain English.")

EXAMPLES = [
    "Which deals are overdue?",
    "What is the total value in Proposal stage?",
    "Which deals should I chase first?",
    "Show leads with missing contact details",
]
cols = st.columns(len(EXAMPLES))
for i, (col, q) in enumerate(zip(cols, EXAMPLES)):
    if col.button(q, key=f"ex_{i}", width="stretch"):
        st.session_state["pending_q"] = q

typed = st.chat_input("Ask about your pipeline...")
question = typed or st.session_state["pending_q"]
st.session_state["pending_q"] = None

if question:
    # 1. Add user question to chat
    st.session_state["chat"].append({"role": "user", "content": question})
    
    # 2. Call AI (pass clean data, not flagged, so duplicates are already handled)
    with st.spinner("Thinking..."):
        answer = ai.answer_question(question, clean)
        
    # 3. Add AI response dict to chat
    st.session_state["chat"].append({"role": "assistant", "content": answer})

# 4. Render chat history
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], dict):
            # Render the text
            st.markdown(msg["content"]["text"])
            
            # Render the table if it has rows
            if not msg["content"]["table"].empty:
                st.dataframe(msg["content"]["table"], use_container_width=True)
                
            # Render the filter expander
            with st.expander("How I understood this"):
                st.json(msg["content"]["filters"])
        else:
            st.markdown(msg["content"])

if st.session_state["chat"]:
    if st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()
else:
    st.info("Pick an example above or type your own question.")