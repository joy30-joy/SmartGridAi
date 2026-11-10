# ⚡ SmartGrid AI | Agentic AI & RAG Energy Intelligence Platform

An end-to-end AI-powered electricity demand forecasting, tool-calling feature extraction, and consumer advisory platform for the Texas ERCOT electrical grid.

---

## 🌟 Highlights & New Features

### 1. 🤖 Agentic AI Feature Extractor (`agentic_ai.py`)
- **Autonomous Tool Calling**: Leverages an LLM agent with the function/tool `derive(temperature_mean, temperature_max, temperature_min, date_iso, ercot_zone)`.
- **Feature Engineering Engine**: Automatically derives:
  - **`CDD` (Cooling Degree Days)**: $\max(0, \text{temperature\_mean} - 18.3)$
  - **`temp_range`**: $\text{temperature\_max} - \text{temperature\_min}$
  - **`weekend`**: Boolean flag indicating if the target date falls on Saturday or Sunday.
  - **`ERCOT_zone_encoded`**: Numerical category index based on ERCOT zones (`COAST`, `EAST`, `NORTH`, `SOUTH`, `WEST`).
- **Natural Language Parsing**: Accepts raw text reports such as:
  ```
  "Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW"
  ```
  and returns the structured JSON:
  ```json
  {
    "zone_usage": 52000.0,
    "temperature_mean": 29.0,
    "temperature_max": 35.0,
    "temperature_min": 24.0,
    "humidity": 60.0,
    "precipitation_mm": 0.0,
    "wind_speed_mps": 4.0,
    "snowfall_cm": null,
    "CDD": 10.7,
    "temp_range": 11.0,
    "weekend": 1,
    "ERCOT_zone_encoded": 2
  }
  ```
- **Zero-Failure Fallback Engine**: Works seamlessly with Groq API (`GROQ_API_KEY`), Google Gemini API (`GEMINI_API_KEY`), OpenAI API, or an integrated deterministic NLP extraction engine if running offline or without an API key.

---

### 2. 💬 Premier SmartGrid Chatbot (`app.py`)
- **Conversational Energy Forecasting**: Users can chat naturally, enter any city/date/weather description, or click pre-built scenario chips (Dallas Heatwave, Houston Summer Peak, Austin Winter Freeze).
- **Inline Agentic Breakdown**: Shows real-time derived parameters (`CDD`, `temp_range`, `weekend`, `ERCOT_zone_encoded`).
- **Live ML Forecasting**: Predicts electricity demand in Megawatts (MW), high-usage risk classification, and probability.
- **RAG-Grounded Advisory**: Retrieves relevant excerpts from the 8 PDF knowledge base documents (`documents/`) and cites sources with document names and page numbers.
- **Dashboard Synchronization**: One-click integration to sync scenarios between the Chatbot and the Forecaster Dashboard.

---

### 3. 📊 Full Forecaster Dashboard & Studio
- **Preserved Dashboard**: Location selection (COAST, NORTH, SOUTH, WEST), date picker, weather sliders, high-usage risk alerts, and usage vs. threshold comparative charts.
- **Agentic AI Inspector Tab**: Live interactive developer studio demonstrating LLM function calling and JSON output generation.

---

## 📁 Project Structure

```
SmartGrid-AI/
├── artifacts/
│   ├── usage_classification_model.pkl   # Trained High-Usage Classifier (XGBoost/RF)
│   └── usage_regression_model.pkl       # Trained Demand Regression Model
├── data/
│   ├── texas_outages_synthetic_50000-5.csv
│   └── texas_synthetic_heat_sensitive_50000.csv  # 50k weather & load records
├── documents/                            # 8 Technical PDF Reference Documents
│   ├── 01_Texas_ERCOT_Grid_Overview.pdf
│   ├── 02_Weather_Impacts_on_Electricity_Demand.pdf
│   ├── 03_Power_Outages_Causes_and_Impacts.pdf
│   ├── 04_Grid_Stress_Reserves_and_Emergency_Operations.pdf
│   ├── 05_Demand_Forecasting_and_Feature_Engineering.pdf
│   ├── 06_Consumer_Outage_Preparedness_and_Advisories.pdf
│   ├── 07_RAG_Knowledge_Retrieval_and_Grounded_Explanations.pdf
│   └── 08_SmartGrid_AI_Architecture_and_Risk_Communication.pdf
├── ml/
│   ├── classification.py                # Model training pipeline
│   ├── regression.py                    # Demand regression pipeline
│   ├── prediction.py                    # Inference engine
│   └── eda.py                           # Exploratory data analysis
├── rag/
│   ├── chunking.py                      # Text chunking with overlap
│   ├── embeddings.py                    # Sentence Transformers (MiniLM-L6-v2)
│   ├── ingestion.py                     # PyMuPDF document loader
│   ├── llm.py                           # Gemini / Groq / Grounded advisory generator
│   ├── retrieval.py                     # Cosine similarity retrieval
│   └── smartgrid_pipeline.py            # End-to-end RAG pipeline
├── agentic_ai.py                        # Agentic AI Feature Extractor & Tool Calling
├── app.py                               # Streamlit UI (Chatbot + Dashboard + Studio)
├── requirements.txt                     # Dependencies
├── test_agentic_ai.py                   # Automated tests for Agentic AI
├── test_prediction.py                   # ML Model tests
└── test_smartgrid.py                    # Complete Pipeline test
```

---

## 🚀 Quickstart Guide

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Agentic AI CLI Tool
Run the autonomous feature extractor directly from terminal:
```bash
python agentic_ai.py
```
**Example Input:**
```
Describe conditions: Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW
```

### 3. Launch the Streamlit Web Application
Launch the complete Energy Intelligence & Chatbot platform:
```bash
streamlit run app.py
```

### 4. Run Automated Verification Tests
```bash
python test_agentic_ai.py
python test_smartgrid.py
```

---

## 🔑 Environment Variables (Optional)
Create a `.env` file in the project root:
```env
# Optional: For Agentic AI LLM Tool Calling (defaults to robust offline extractor if omitted)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Optional: For Gemini-powered RAG Advisories (defaults to grounded PDF synthesizer if omitted)
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 📊 Dataset Alignment
Features extracted by the Agentic AI directly align with `texas_synthetic_heat_sensitive_50000.csv`:
| Extracted Feature | Dataset Field | Description |
|---|---|---|
| `temperature_mean` | `temperature_mean` | Average 24-hr temperature in °C |
| `temperature_max` | `temperature_max` | Daily maximum temperature in °C |
| `temperature_min` | `temperature_min` | Daily minimum temperature in °C |
| `CDD` | `CDD` | Cooling Degree Days: $\max(0, T_{\text{mean}} - 18.3)$ |
| `temp_range` | Derived | Diurnal range: $T_{\text{max}} - T_{\text{min}}$ |
| `weekend` | Derived | Weekend flag: 1 (Sat/Sun), 0 (Weekday) |
| `ERCOT_zone_encoded` | `ERCOT_zone` | Zone index (COAST=0, EAST=1, NORTH=2, SOUTH=3, WEST=4) |
| `zone_usage` | `zone_usage` | Regional electricity demand in MW |
