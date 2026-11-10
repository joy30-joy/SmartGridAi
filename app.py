import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import date, datetime

# Internal imports
from agentic_ai import (
    run as run_agentic_ai,
    to_pipeline_input,
    derive as derive_tool,
    ZONES,
    ZONE_DEFAULTS
)
from ml.prediction import predict_grid_status
from rag.smartgrid_pipeline import run_smartgrid, load_knowledge_base
from rag.retrieval import retrieve_documents
from rag.llm import generate_answer

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartGrid AI | Energy Intelligence & Agentic Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111c30;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 12px;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-top: -6px;
    margin-bottom: 12px;
}

.badge-tag {
    background-color: #172840;
    color: #38bdf8;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #263a55;
    display: inline-block;
}

.section-title {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 10px;
}

.metric-box {
    background: linear-gradient(135deg, #172840, #142033);
    border: 1px solid #263a55;
    padding: 20px;
    border-radius: 14px;
    min-height: 125px;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 750;
    margin-top: 6px;
}

.metric-note {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 4px;
}

.status-high {
    background: linear-gradient(135deg, #451c29, #2b1119);
    color: #fca5a5;
    border: 1px solid #9f3349;
    padding: 16px 20px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-normal {
    background: linear-gradient(135deg, #12352c, #0a201b);
    color: #86efac;
    border: 1px solid #267653;
    padding: 16px 20px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.agent-card {
    background-color: #121e33;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    margin-top: 10px;
    margin-bottom: 14px;
}

.agent-pill {
    background-color: #182b47;
    border: 1px solid #2b4c77;
    color: #bae6fd;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-family: monospace;
    display: inline-block;
    margin: 3px 4px 3px 0px;
}

.stButton > button {
    background-color: #087fce;
    color: white;
    border-radius: 9px;
    border: none;
    height: 44px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #0b9bf0;
    color: white;
    box-shadow: 0 4px 12px rgba(8, 127, 206, 0.3);
}

.chip-btn > button {
    background-color: #152238 !important;
    border: 1px solid #263a55 !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
    height: 38px !important;
    border-radius: 20px !important;
}

.chip-btn > button:hover {
    background-color: #1e3557 !important;
    color: #38bdf8 !important;
    border-color: #38bdf8 !important;
}

div[data-testid="stExpander"] {
    background-color: #111c30;
    border: 1px solid #263a55;
    border-radius: 10px;
}

hr {
    border-color: #263a55;
}

/* Chat bubble styling */
.user-msg {
    background-color: #172c4c;
    border: 1px solid #294670;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin-bottom: 12px;
    color: #f1f5f9;
}

.assistant-msg {
    background-color: #121c2e;
    border: 1px solid #20334e;
    border-radius: 12px 12px 12px 2px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #e2e8f0;
}

/* ==========================================
   INPUT BOXES & TYPING TEXT VISIBILITY FIX
   ========================================== */

/* Universal input container styling */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
    background-color: #152238 !important;
    border: 1.5px solid #2d456b !important;
    border-radius: 9px !important;
    color: #ffffff !important;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="textarea"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
    background-color: #1a2a44 !important;
}

/* Text inputs, number inputs, textareas, date inputs */
input,
textarea,
input[type="text"],
input[type="number"],
input[type="password"],
input[type="date"] {
    background-color: #152238 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    caret-color: #38bdf8 !important;
}

/* Inner inputs inside BaseWeb wrappers */
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #38bdf8 !important;
}

/* Placeholders */
input::placeholder,
textarea::placeholder,
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="textarea"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    opacity: 0.9 !important;
}

/* Number input increment/decrement buttons */
div[data-testid="stNumberInput"] button {
    background-color: #1e3352 !important;
    color: #ffffff !important;
    border: 1px solid #2d456b !important;
}

div[data-testid="stNumberInput"] button:hover {
    background-color: #2b4975 !important;
    color: #38bdf8 !important;
}

/* Labels above inputs */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}

/* ==========================================
   STREAMLIT CHAT INPUT & PROMPT BOX FIX
   ========================================== */

/* Main chat input container */
div[data-testid="stChatInput"] {
    background-color: transparent !important;
    padding-bottom: 8px !important;
}

div[data-testid="stChatInput"] > div {
    background-color: #152238 !important;
    border: 1.5px solid #2d456b !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
}

div[data-testid="stChatInput"] > div:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3), 0 4px 16px rgba(0, 0, 0, 0.45) !important;
    background-color: #1a2a44 !important;
}

/* Chat input textarea itself */
div[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    caret-color: #38bdf8 !important;
    line-height: 1.5 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

/* Chat input send button */
div[data-testid="stChatInput"] button {
    color: #38bdf8 !important;
    background-color: transparent !important;
    border: none !important;
}

div[data-testid="stChatInput"] button:hover {
    color: #7dd3fc !important;
    background-color: #1e3352 !important;
}

div[data-testid="stChatInput"] button svg {
    fill: #38bdf8 !important;
}

/* Selectbox dropdown options menu */
div[data-baseweb="select"] * {
    color: #ffffff !important;
}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #152238 !important;
    border: 1px solid #2d456b !important;
    border-radius: 8px !important;
}

li[role="option"] {
    color: #e2e8f0 !important;
    background-color: #152238 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "last_input" not in st.session_state:
    st.session_state.last_input = {
        "latitude": 32.75,
        "longitude": -96.77,
        "ERCOT_zone": "NORTH",
        "date": date.today().isoformat(),
        "temperature_mean": 25.0,
        "temperature_max": 32.0,
        "temperature_min": 18.0,
        "precipitation_mm": 0.0,
        "wind_speed_mps": 5.0,
        "snowfall_cm": 0.0,
        "humidity": 60.0
    }

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am **SmartGrid AI Assistant**, powered by **Agentic AI & ERCOT Grid Intelligence**.\n\n"
                "You can:\n"
                "1. **Describe conditions in natural language** (e.g. *'Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW'*)\n"
                "2. The **Agentic AI** will extract the features, invoke the `derive` tool (computing Cooling Degree Days, Temp Range, Weekend flag, ERCOT Zone), and forecast grid stress and electricity demand.\n"
                "3. **Ask domain questions** about ERCOT reserves, EEA levels, weather impact, or consumer advisories."
            ),
            "meta": None
        }
    ]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ==========================================
# 4. HEADER
# ==========================================

col_h1, col_h2 = st.columns([0.7, 0.3])
with col_h1:
    st.markdown('<div class="main-title">⚡ SmartGrid AI <span class="badge-tag">Agentic AI & RAG</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Autonomous Electricity Demand Forecasting, Tool-Calling Intelligence & Consumer Advisory Platform</div>', unsafe_allow_html=True)

with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge-tag" style="background-color: #064e3b; color: #6ee7b7; border-color: #047857;">● ML Models Active</span>
        <span class="badge-tag" style="background-color: #1e3a5f; color: #7dd3fc; border-color: #0284c7;">● Agentic Tools Ready</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# 5. SIDEBAR CONTROLS
# ==========================================

st.sidebar.markdown("### ⚙️ System Settings")
st.sidebar.caption("Configure environment, models & manual forecaster.")

# Model & API settings expander
with st.sidebar.expander("🔑 Agent & API Configuration", expanded=False):
    groq_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Used by Agentic AI for tool calling (Groq / OpenAI API compatible)."
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input

    gemini_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Used for RAG grid advisory generation."
    )
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input

    agent_model = st.selectbox(
        "Agentic LLM Model",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    os.environ["GROQ_MODEL"] = agent_model

st.sidebar.divider()

# Manual location & weather inputs
st.sidebar.subheader("📍 Location Settings")
zone = st.sidebar.selectbox("ERCOT Zone", ["COAST", "NORTH", "SOUTH", "WEST"], index=["COAST", "NORTH", "SOUTH", "WEST"].index(st.session_state.last_input.get("ERCOT_zone", "NORTH")))
latitude = st.sidebar.number_input("Latitude", value=float(st.session_state.last_input.get("latitude", 32.75)), format="%.5f")
longitude = st.sidebar.number_input("Longitude", value=float(st.session_state.last_input.get("longitude", -96.77)), format="%.5f")

try:
    init_date = date.fromisoformat(st.session_state.last_input.get("date", date.today().isoformat()))
except Exception:
    init_date = date.today()

selected_date = st.sidebar.date_input("Prediction Date", value=init_date)

st.sidebar.divider()
st.sidebar.subheader("🌦️ Weather Conditions")

temperature_mean = st.sidebar.number_input("Average Temperature (°C)", value=float(st.session_state.last_input.get("temperature_mean", 25.0)), step=1.0)
temperature_max = st.sidebar.number_input("Maximum Temperature (°C)", value=float(st.session_state.last_input.get("temperature_max", 32.0)), step=1.0)
temperature_min = st.sidebar.number_input("Minimum Temperature (°C)", value=float(st.session_state.last_input.get("temperature_min", 18.0)), step=1.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=int(st.session_state.last_input.get("humidity", 60)))
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, value=float(st.session_state.last_input.get("wind_speed_mps", 5.0)))
precipitation = st.sidebar.number_input("Precipitation (mm)", min_value=0.0, value=float(st.session_state.last_input.get("precipitation_mm", 0.0)))
snowfall = st.sidebar.number_input("Snowfall (cm)", min_value=0.0, value=float(st.session_state.last_input.get("snowfall_cm", 0.0)))

st.sidebar.divider()
predict_button = st.sidebar.button("⚡ Run Manual Forecast", type="primary", use_container_width=True)

# Update sidebar prediction action
if predict_button:
    if temperature_min > temperature_mean:
        st.sidebar.error("Minimum temperature cannot exceed average temperature.")
    elif temperature_mean > temperature_max:
        st.sidebar.error("Average temperature cannot exceed maximum temperature.")
    else:
        manual_input = {
            "latitude": latitude,
            "longitude": longitude,
            "ERCOT_zone": zone,
            "date": selected_date.isoformat(),
            "temperature_mean": temperature_mean,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "precipitation_mm": precipitation,
            "wind_speed_mps": wind_speed,
            "snowfall_cm": snowfall,
            "humidity": humidity
        }
        with st.spinner("Analyzing energy demand and retrieving grid knowledge..."):
            try:
                res = run_smartgrid(manual_input)
                st.session_state.prediction_result = res
                st.session_state.last_input = manual_input
                st.sidebar.success("Forecast updated!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# ==========================================
# 6. MAIN NAVIGATION TABS
# ==========================================

tab_chat, tab_dashboard, tab_inspector = st.tabs([
    "💬 SmartGrid AI Chatbot & Agent",
    "📊 Forecaster Dashboard",
    "🛠️ Agentic AI Inspector (Studio)"
])

# -------------------------------------------------------------
# TAB 1: SMARTGRID AI CHATBOT
# -------------------------------------------------------------
with tab_chat:
    st.markdown("#### ⚡ Conversational Energy Intelligence")
    st.caption("Ask questions or describe weather & grid scenarios in natural language. The Agentic AI calls tools to extract and forecast demand.")

    # Quick prompt chips
    st.markdown("**💡 Quick Scenarios (Click to test):**")
    chip_cols = st.columns(4)
    with chip_cols[0]:
        if st.button("🔥 Dallas Heatwave (Nov 2026)", key="chip_dallas", use_container_width=True):
            st.session_state.pending_prompt = "Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW"
    with chip_cols[1]:
        if st.button("🌊 Houston Summer Coastal Peak", key="chip_houston", use_container_width=True):
            st.session_state.pending_prompt = "July 22 2026, Houston, high 40C, low 29C, avg 34C, humidity 82%, wind 3 m/s, rain 0mm, load 68000 MW"
    with chip_cols[2]:
        if st.button("❄️ Austin Winter Freeze", key="chip_austin", use_container_width=True):
            st.session_state.pending_prompt = "Jan 18 2026, Austin, high 2C, low -7C, avg -3C, humidity 65%, wind 10 m/s, snowfall 3cm, load 62000 MW"
    with chip_cols[3]:
        if st.button("❓ ERCOT EEA Alert Rules", key="chip_eea", use_container_width=True):
            st.session_state.pending_prompt = "What are the ERCOT Energy Emergency Alert (EEA) levels, reserve thresholds, and consumer actions?"

    st.write("")

    # Display Chat History
    for msg_idx, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg"><b>👤 You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-msg"><b>⚡ SmartGrid AI:</b><br>', unsafe_allow_html=True)
            st.markdown(msg["content"])

            # If message contains structured meta (from an Agentic execution)
            if msg.get("meta"):
                meta = msg["meta"]
                ext = meta.get("extracted", {})
                pred = meta.get("prediction", {})

                # Render Agentic Derivation Badges
                st.markdown(f"""
                <div class="agent-card">
                    <div style="font-size: 13px; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">
                        🤖 AGENTIC AI EXTRACTION & DERIVED FEATURES (TOOL CALL: <code>derive()</code>)
                    </div>
                    <div>
                        <span class="agent-pill">📅 Date: {ext.get('date', 'Unknown')}</span>
                        <span class="agent-pill">📍 Zone: {ext.get('zone', 'NORTH')}</span>
                        <span class="agent-pill">🌡️ Mean: {ext.get('temperature_mean')}°C</span>
                        <span class="agent-pill">🔥 Max: {ext.get('temperature_max')}°C</span>
                        <span class="agent-pill">❄️ Min: {ext.get('temperature_min')}°C</span>
                        <span class="agent-pill">📊 CDD: {ext.get('CDD')}</span>
                        <span class="agent-pill">↔️ Temp Range: {ext.get('temp_range')}°C</span>
                        <span class="agent-pill">🗓️ Weekend: {'Yes' if ext.get('weekend') == 1 else 'No'}</span>
                        <span class="agent-pill">🔢 Zone Encoded: {ext.get('ERCOT_zone_encoded')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Prediction metric summary
                if pred:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Predicted Usage", f"{pred['predicted_usage']:,.2f} MW")
                    with col_m2:
                        st.metric("Usage Threshold", f"{pred['usage_threshold']:,.2f} MW")
                    with col_m3:
                        st.metric("High Usage Probability", f"{pred['risk_probability']:.2f}%")

                    if ext.get("zone_usage"):
                        delta = ext["zone_usage"] - pred["predicted_usage"]
                        st.info(f"📊 **Reported Load:** {ext['zone_usage']:,.0f} MW  |  **Model Forecast:** {pred['predicted_usage']:,.0f} MW  |  **Variance:** {delta:+,.0f} MW")

                # Show sources if any
                sources = meta.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Knowledge Sources ({len(sources)} documents cited)"):
                        for s in sources:
                            st.write(f"- **{s.get('document')}** (Page {s.get('page')}) — Similarity: `{s.get('similarity_score', 0):.4f}`")

            st.markdown('</div>', unsafe_allow_html=True)

    # Process incoming or pending prompt
    chat_input_val = st.chat_input("Ask a question or describe conditions (e.g. Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, load 52000 MW)...")
    prompt_to_process = chat_input_val or st.session_state.pending_prompt

    if prompt_to_process:
        st.session_state.pending_prompt = None

        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt_to_process,
            "meta": None
        })

        # Process message
        with st.spinner("Agentic AI analyzing scenario and synthesizing grid intelligence..."):
            lower_p = prompt_to_process.lower()
            # Detect if message contains weather or grid conditions
            is_condition_query = any(k in lower_p for k in [
                "high", "low", "avg", "temp", "celsius", "humidity", "mw", "load", "wind", "rain", "dallas", "houston", "austin", "ercot", "zone", "weather"
            ]) and any(char.isdigit() for char in prompt_to_process)

            if is_condition_query:
                # 1. Run Agentic AI Feature Extractor
                extracted_json = run_agentic_ai(prompt_to_process)
                pipeline_in = to_pipeline_input(extracted_json, prompt_to_process)

                # 2. Run ML Prediction
                pred_result = predict_grid_status(pipeline_in)

                # 3. Retrieve Documents
                try:
                    chunks, embeddings = load_knowledge_base()
                    retrieved = retrieve_documents(pred_result["rag_query"], chunks, embeddings, top_k=3)
                    advisory = generate_answer(pred_result["rag_query"], retrieved)
                    sources_list = [
                        {
                            "document": item["metadata"]["source"],
                            "page": item["metadata"]["page"],
                            "similarity_score": round(float(item["score"]), 4)
                        }
                        for item in retrieved
                    ]
                except Exception as ex:
                    advisory = f"ML Prediction completed successfully. RAG Retrieval note: {ex}"
                    sources_list = []

                # Update session state for dashboard sync
                st.session_state.prediction_result = {
                    "predicted_usage": pred_result["predicted_usage"],
                    "risk_status": pred_result["risk_status"],
                    "risk_probability": pred_result["risk_probability"],
                    "usage_threshold": pred_result["usage_threshold"],
                    "explanation": advisory,
                    "sources": sources_list
                }
                st.session_state.last_input = pipeline_in

                # Construct assistant reply text
                risk_emoji = "⚠️ HIGH USAGE RISK" if "high" in pred_result["risk_status"].lower() else "✅ NORMAL USAGE"
                reply_text = f"""### ⚡ SmartGrid AI Assessment: **{risk_emoji}**

**Forecast Summary:**
- **Predicted Demand:** **{pred_result['predicted_usage']:,.2f} MW**
- **Threshold Limit:** **{pred_result['usage_threshold']:,.2f} MW**
- **Risk Probability:** **{pred_result['risk_probability']:.2f}%**
- **Classification Status:** `{pred_result['risk_status']}`

---
{advisory}
"""
                meta_dict = {
                    "extracted": {
                        "date": pipeline_in["date"],
                        "zone": pipeline_in["ERCOT_zone"],
                        "temperature_mean": extracted_json.get("temperature_mean"),
                        "temperature_max": extracted_json.get("temperature_max"),
                        "temperature_min": extracted_json.get("temperature_min"),
                        "CDD": extracted_json.get("CDD"),
                        "temp_range": extracted_json.get("temp_range"),
                        "weekend": extracted_json.get("weekend"),
                        "ERCOT_zone_encoded": extracted_json.get("ERCOT_zone_encoded"),
                        "zone_usage": extracted_json.get("zone_usage")
                    },
                    "prediction": pred_result,
                    "sources": sources_list
                }

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": reply_text,
                    "meta": meta_dict
                })

            else:
                # General Question: Query RAG directly
                try:
                    chunks, embeddings = load_knowledge_base()
                    retrieved = retrieve_documents(prompt_to_process, chunks, embeddings, top_k=3)
                    advisory = generate_answer(prompt_to_process, retrieved)
                    sources_list = [
                        {
                            "document": item["metadata"]["source"],
                            "page": item["metadata"]["page"],
                            "similarity_score": round(float(item["score"]), 4)
                        }
                        for item in retrieved
                    ]
                except Exception as ex:
                    advisory = f"I am unable to access the PDF knowledge base currently ({ex})."
                    sources_list = []

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": advisory,
                    "meta": {"sources": sources_list} if sources_list else None
                })

        st.rerun()

    # Chat reset buttons
    col_c1, col_c2 = st.columns([0.8, 0.2])
    with col_c2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

# -------------------------------------------------------------
# TAB 2: FORECASTER DASHBOARD (PRESERVED FULL INTERACTIVE VIEW)
# -------------------------------------------------------------
with tab_dashboard:
    result = st.session_state.prediction_result
    input_data = st.session_state.last_input

    if result is None:
        st.markdown('<div class="section-title">📊 Energy Intelligence Dashboard</div>', unsafe_allow_html=True)
        st.write("Configure location and weather conditions in the sidebar or run an analysis in the Chatbot to view demand forecast analytics.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">MACHINE LEARNING</div>
                <div class="metric-value">Demand Forecast</div>
                <div class="metric-note">XGBoost & Regression modeling</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">RISK ANALYTICS</div>
                <div class="metric-value">Risk Detection</div>
                <div class="metric-note">Classification probability & thresholds</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">AGENTIC AI + RAG</div>
                <div class="metric-value">Tool Derivation</div>
                <div class="metric-note">CDD, weekend flag, grounded advisories</div>
            </div>
            """, unsafe_allow_html=True)

        st.info("💡 Tip: You can test the Dallas scenario directly from the **💬 SmartGrid AI Chatbot** tab or click 'Run Manual Forecast' in the sidebar!")

    else:
        st.markdown('<div class="section-title">📈 Prediction Overview</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">PREDICTED ELECTRICITY USAGE</div>
                <div class="metric-value">{result['predicted_usage']:,.2f} MW</div>
                <div class="metric-note">Model regression forecast</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">HIGH-USAGE PROBABILITY</div>
                <div class="metric-value">{result['risk_probability']:.2f}%</div>
                <div class="metric-note">Classifier stress probability</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">USAGE THRESHOLD</div>
                <div class="metric-value">{result['usage_threshold']:,.2f} MW</div>
                <div class="metric-note">75th percentile stress boundary</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Risk Assessment Banner
        st.markdown('<div class="section-title">🚦 Risk Assessment</div>', unsafe_allow_html=True)
        risk_status = str(result.get("risk_status", "")).strip().lower()
        is_high_risk = risk_status in ["high usage risk", "high usage", "high risk"]

        if is_high_risk:
            st.markdown("""
            <div class="status-high">
                ⚠️ HIGH USAGE RISK DETECTED — Elevated strain anticipated on ERCOT transmission infrastructure.
            </div>
            """, unsafe_allow_html=True)
            st.warning("The model has identified extreme thermal or demand conditions exceeding safe threshold limits.")
        else:
            st.markdown("""
            <div class="status-normal">
                ✅ NORMAL USAGE CLASSIFICATION — Electricity demand is projected within standard operating reserves.
            </div>
            """, unsafe_allow_html=True)
            st.success("The model has classified conditions within normal reserve margins.")

        # Usage vs Threshold Comparison
        st.divider()
        st.markdown('<div class="section-title">📊 Usage vs Threshold Comparison</div>', unsafe_allow_html=True)

        usage_df = pd.DataFrame({
            "Category": ["Predicted Usage", "High Usage Threshold"],
            "Value": [result["predicted_usage"], result["usage_threshold"]]
        })

        st.bar_chart(usage_df, x="Category", y="Value", horizontal=True)

        difference = result["predicted_usage"] - result["usage_threshold"]
        if difference > 0:
            st.write(f"Predicted usage is **{difference:,.2f} MW** above the high-usage threshold.")
        else:
            st.write(f"Predicted usage is **{abs(difference):,.2f} MW** below the high-usage threshold.")

        # Weather & Input Conditions Summary
        st.divider()
        st.markdown('<div class="section-title">🌦️ Input Conditions Summary</div>', unsafe_allow_html=True)
        w_c1, w_c2, w_c3, w_c4 = st.columns(4)
        with w_c1:
            st.metric("Average Temperature", f"{input_data['temperature_mean']} °C")
        with w_c2:
            st.metric("Humidity", f"{input_data['humidity']}%")
        with w_c3:
            st.metric("Wind Speed", f"{input_data['wind_speed_mps']} m/s")
        with w_c4:
            st.metric("Precipitation", f"{input_data['precipitation_mm']} mm")

        st.caption(f"**ERCOT Zone:** {input_data.get('ERCOT_zone')} | **Date:** {input_data.get('date')} | **Coordinates:** {input_data.get('latitude')}, {input_data.get('longitude')}")

        # AI-Powered Grid Advisory
        st.divider()
        st.markdown('<div class="section-title">🤖 AI-Powered Grid Advisory</div>', unsafe_allow_html=True)
        st.markdown(result["explanation"])

        # Knowledge Retrieval Sources
        if result.get("sources"):
            st.divider()
            st.markdown('<div class="section-title">📚 Knowledge Retrieval Sources</div>', unsafe_allow_html=True)
            st.caption("Retrieved ERCOT technical documents supporting the generated AI explanation.")

            for idx, src in enumerate(result["sources"], start=1):
                with st.expander(f"📄 Source {idx}: {src.get('document')} (Page {src.get('page')})"):
                    st.write("**Document:**", src.get("document"))
                    st.write("**Page:**", src.get("page"))
                    st.write("**Similarity Score:**", f"{src.get('similarity_score', 0):.4f}")

# -------------------------------------------------------------
# TAB 3: AGENTIC AI INSPECTOR (DEVELOPER / FUNCTION CALLING STUDIO)
# -------------------------------------------------------------
with tab_inspector:
    st.markdown("#### 🛠️ Agentic AI Function Calling Studio")
    st.caption("Inspect the exact tool-calling workflow, function parameter extraction (`derive`), and output JSON matching the project dataset.")

    test_prompt = st.text_area(
        "Condition Description Text:",
        value="Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW",
        height=90
    )

    if st.button("🚀 Execute Agentic Tool Calling", type="primary"):
        with st.spinner("Invoking Agentic AI & Tool Pipeline..."):
            extracted_output = run_agentic_ai(test_prompt)

            st.success("Agentic Tool Execution Completed!")

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("**📤 Extracted Features JSON (User Required Keys):**")
                st.json(extracted_output)

            with col_t2:
                st.markdown("**⚙️ Tool Call Diagnostics (`derive`):**")
                st.markdown(f"""
                - **Cooling Degree Days (CDD):** `{extracted_output.get('CDD')}` (Derived as max(0, Mean Temp - 18.3°C))
                - **Temperature Range:** `{extracted_output.get('temp_range')}` °C (High - Low)
                - **Weekend Flag:** `{extracted_output.get('weekend')}` (1 = Weekend, 0 = Weekday)
                - **ERCOT Zone Encoded:** `{extracted_output.get('ERCOT_zone_encoded')}` (Zone index in `{ZONES}`)
                - **Reported Load:** `{extracted_output.get('zone_usage')}` MW
                - **Precipitation:** `{extracted_output.get('precipitation_mm')}` mm
                - **Wind Speed:** `{extracted_output.get('wind_speed_mps')}` m/s
                """)

                # Show pipeline mapping
                pip_in = to_pipeline_input(extracted_output, test_prompt)
                st.markdown("**🗺️ Dataset Feature Alignment:**")
                st.code(json.dumps(pip_in, indent=2), language="json")

# ==========================================
# 7. FOOTER
# ==========================================

st.divider()
st.caption("⚡ SmartGrid AI | Autonomous Energy Demand Forecasting, Agentic AI & Consumer Advisory Platform")
st.caption("© 2026 SmartGrid AI. All rights reserved.")
