import os

import streamlit as st

from paperpilot.config import settings
from paperpilot.llm import LLMError
from paperpilot.service import PaperPilot

st.set_page_config(page_title="PaperPilot", layout="wide")


@st.cache_resource
def get_app() -> PaperPilot:
    return PaperPilot()


pp = get_app()
st.session_state.setdefault("history", [])
st.session_state.setdefault("indexed", set())


def show_sources(hits, invalid=()):
    if invalid:
        st.warning(f"Model cited passage(s) {list(invalid)} that were not retrieved. Verify manually.")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        label = f"[{i}] {m['title'][:48]} | {m['section']} | p.{m['page']} | score {h['score']:.2f}"
        with st.expander(label):
            st.write(h["text"])


# ------------------------------- sidebar: library -------------------------------
with st.sidebar:
    st.header("Library")
    for f in st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True) or []:
        if f.name in st.session_state.indexed:
            continue
        if f.size > settings.max_upload_mb * 1024 * 1024:
            st.error(f"{f.name}: larger than {settings.max_upload_mb} MB.")
            st.session_state.indexed.add(f.name)
            continue
        with st.spinner(f"Indexing {f.name}..."):
            os.makedirs(settings.upload_dir, exist_ok=True)
            path = os.path.join(settings.upload_dir, os.path.basename(f.name))
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            try:
                pp.add_pdf(path)
            except ValueError as e:
                st.error(f"{f.name}: {e}")
            st.session_state.indexed.add(f.name)

    papers = pp.store.list_papers()
    titles = {p["paper_id"]: p["title"] for p in papers}
    st.caption("Search scope")
    selected = [p["paper_id"] for p in papers
                if st.checkbox(p["title"][:55], value=True, key=p["paper_id"])]

st.title("PaperPilot")
tab_ask, tab_sum, tab_cmp = st.tabs(["Ask", "Summarize", "Compare"])

# ------------------------------------ Ask ---------------------------------------
with tab_ask:
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)
    if q := st.chat_input("Ask about the selected papers..."):
        prior = list(st.session_state.history)
        st.session_state.history.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and reading..."):
                    res = pp.ask(q, selected or None, history=prior)
                st.markdown(res["answer"])
                show_sources(res["sources"], res["invalid_citations"])
                st.session_state.history.append(("assistant", res["answer"]))
            except LLMError as e:
                st.error(str(e))
                st.session_state.history.pop()

# ---------------------------------- Summarize -----------------------------------
with tab_sum:
    if not papers:
        st.info("Upload a paper in the sidebar to begin.")
    else:
        pid = st.selectbox("Paper", list(titles), format_func=lambda x: titles[x])
        if st.button("Generate summary"):
            try:
                with st.spinner("Reading the paper..."):
                    out = pp.summarize(pid)
                st.subheader("TL;DR")
                st.write(out["tldr"])
                for name, text, hits in out["sections"]:
                    st.markdown(f"**{name}**")
                    st.write(text)
                    show_sources(hits)
            except LLMError as e:
                st.error(str(e))

# ----------------------------------- Compare ------------------------------------
with tab_cmp:
    aspects = st.text_input("Aspects (comma-separated)",
                            "problem, method, datasets, results, limitations")
    if len(selected) < 2:
        st.info("Select at least two papers in the sidebar.")
    elif st.button("Compare papers"):
        try:
            with st.spinner("Comparing..."):
                res = pp.compare(selected[:5], [a.strip() for a in aspects.split(",") if a.strip()])
            st.markdown(res["answer"])
            show_sources(res["sources"], res["invalid_citations"])
        except LLMError as e:
            st.error(str(e))
