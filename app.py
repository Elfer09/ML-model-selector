import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from modules.data_processor import load_and_profile
from modules.llm_analyst import ask_llm
from modules.visualizer import auto_charts
from modules.ml_trainer import train_and_compare

load_dotenv()  # loads .env locally; no-op on Streamlit Cloud

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        placeholder="sk-..." if not env_key else "Loaded from environment ✓",
    )
    if env_key:
        st.caption("✅ Key loaded from environment.")
    else:
        st.caption("Your key is never stored or sent anywhere except OpenAI's API.")
    st.divider()
    st.markdown("**Mode**")
    app_mode = st.radio(
        "Choose what you want to do:",
        ["🔍 Chat with Your Data", "📊 Auto Visualize", "🤖 Train ML Models"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "Built by [Eleni](https://github.com/Elfer09) · "
        "[Source](https://github.com/Elfer09/ML-model-selector)"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 AI Data Analyst")
st.markdown(
    "Upload any CSV. Ask questions in plain English. Get instant ML-powered insights."
)
st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload a CSV to get started. No data? Try a dataset from [Kaggle](https://www.kaggle.com/datasets).")
    st.stop()

# ── Load + Profile ─────────────────────────────────────────────────────────────
df, profile = load_and_profile(uploaded_file)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{profile['rows']:,}")
col2.metric("Columns", profile['cols'])
col3.metric("Missing Values", f"{profile['missing_pct']:.1f}%")
col4.metric("Numeric Columns", profile['numeric_cols'])

with st.expander("📋 Preview Data (first 100 rows)", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

with st.expander("📈 Statistical Summary", expanded=False):
    st.dataframe(df.describe(include="all").T, use_container_width=True)

st.divider()

# ── Mode: Chat with Your Data ──────────────────────────────────────────────────
if app_mode == "🔍 Chat with Your Data":
    st.subheader("🔍 Chat with Your Data")

    if not api_key:
        st.warning("Add your OpenAI API key in the sidebar to enable AI analysis.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick-start suggestion buttons
    if not st.session_state.messages:
        st.markdown("**Quick questions to try:**")
        suggestions = [
            "What are the most important patterns in this dataset?",
            "Are there any outliers or data quality issues I should know about?",
            "Which columns are most correlated with each other?",
            "Give me a 3-bullet executive summary of this data.",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    # Handle pending question from button click
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                response = ask_llm(question, df, profile, api_key)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Chat input
    if prompt := st.chat_input("Ask anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = ask_llm(prompt, df, profile, api_key)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ── Mode: Auto Visualize ───────────────────────────────────────────────────────
elif app_mode == "📊 Auto Visualize":
    st.subheader("📊 Auto Visualize")
    st.caption("Select columns and chart type. Charts are generated automatically.")
    auto_charts(df)

# ── Mode: Train ML Models ──────────────────────────────────────────────────────
elif app_mode == "🤖 Train ML Models":
    st.subheader("🤖 Train & Compare ML Models")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns to train ML models.")
        st.stop()

    col_left, col_right = st.columns(2)
    with col_left:
        target = st.selectbox("🎯 Target Column (what to predict)", numeric_cols)
    with col_right:
        task_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)

    features = [c for c in numeric_cols if c != target]
    st.caption(f"Features used: `{', '.join(features)}`")

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training models... this may take a moment."):
            results_df, best_model_name, fig = train_and_compare(df, features, target, task_type)

        st.success(f"✅ Best model: **{best_model_name}**")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results_df, use_container_width=True)
