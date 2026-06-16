import os
import pandas as pd
import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_groq import ChatGroq

# ==========================
# PAGE
# ==========================

st.title("AI Data Analyst Chatbot")
st.markdown("Demo chatbot oleh Flo")

# ==========================
# API KEY
# ==========================

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

if st.session_state["api_key"] == "":

    input_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    submit_key = st.button("Submit Key")

    if submit_key:
        st.session_state["api_key"] = input_api_key
        st.rerun()

    st.stop()

os.environ["GROQ_API_KEY"] = st.session_state["api_key"]

# ==========================
# MODEL
# ==========================

client = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ==========================
# DATASET UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.session_state["df"] = df

    # reset insight jika file berubah

    current_file = uploaded_file.name

    if (
        "uploaded_filename" not in st.session_state
        or st.session_state["uploaded_filename"] != current_file
    ):

        st.session_state["uploaded_filename"] = current_file

        if "auto_insight" in st.session_state:
            del st.session_state["auto_insight"]

    # ==========================
    # PREVIEW
    # ==========================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==========================
    # BASIC INFO
    # ==========================

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # ==========================
    # SUMMARY
    # ==========================

    st.subheader("Dataset Summary")
    st.write(df.describe())

    # ==========================
    # VISUALIZATION
    # ==========================

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:

        st.subheader("Visualization")

        selected_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        st.bar_chart(df[selected_col])

    # ==========================
    # AUTO INSIGHT
    # ==========================

    if "auto_insight" not in st.session_state:

        with st.spinner("Generating AI Insights..."):

            prompt = f"""
            Analyze this dataset.

            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}

            Sample Data:
            {df.head(20).to_string()}

            Provide:
            1. Executive Summary
            2. Key Findings
            3. Trends
            4. Risks
            5. Recommendations

            Use business language.
            """

            insight = client.invoke(prompt)

            st.session_state["auto_insight"] = insight.content

    st.subheader("AI Insights")
    st.write(st.session_state["auto_insight"])

# ==========================
# SYSTEM PROMPT
# ==========================

if "df" in st.session_state:

    df = st.session_state["df"]

    dataset_info = f"""
    Columns:
    {list(df.columns)}

    Shape:
    {df.shape}

    Sample Data:
    {df.head(10).to_string()}
    """

    system_prompt = f"""
    You are a Senior Data Analyst.

    Dataset Information:
    {dataset_info}

    Responsibilities:
    - Answer questions about the dataset
    - Analyze trends
    - Find anomalies
    - Explain findings
    - Provide business recommendations

    Give detailed but easy-to-understand explanations.
    """

else:

    system_prompt = """
    You are a Senior Data Analyst.

    Help users understand their data.
    """

# ==========================
# CHAT HISTORY
# ==========================

if "chat_history" not in st.session_state:

    st.session_state["chat_history"] = [
        SystemMessage(system_prompt)
    ]

# update system prompt

st.session_state["chat_history"][0] = SystemMessage(
    system_prompt
)

# ==========================
# DISPLAY CHAT
# ==========================

for chat in st.session_state["chat_history"]:

    if isinstance(chat, HumanMessage):
        role = "human"

    elif isinstance(chat, SystemMessage):
        continue

    else:
        role = "ai"

    with st.chat_message(role):
        st.markdown(chat.content)

# ==========================
# CHAT INPUT
# ==========================

user_input = st.chat_input(
    "Ask something about your data..."
)

if user_input:

    st.session_state["chat_history"].append(
        HumanMessage(user_input)
    )

    with st.chat_message("human"):
        st.markdown(user_input)

    response = client.invoke(
        st.session_state["chat_history"]
    )

    st.session_state["chat_history"].append(
        AIMessage(response.content)
    )

    st.rerun()
