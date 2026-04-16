# 🤖 AI Data Analyst

> Upload any CSV. Ask questions in plain English. Get instant AI-powered insights, auto-generated charts, and ML model comparisons — all in your browser.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green?logo=openai)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Chat with Your Data** | Ask natural language questions — the LLM answers grounded in your actual dataset |
| 📊 **Auto Visualize** | Build interactive Plotly charts (histogram, scatter, heatmap, box, bar, line) with zero code |
| 🤖 **Train & Compare Models** | Auto-trains 6–7 sklearn models via cross-validation and ranks them instantly |
| 📋 **Data Profiling** | Instant summary: shape, missing values, dtypes, statistical describe |
| 💬 **Chat History** | Persistent conversation within a session with quick-start suggestion buttons |

---

## 🚀 Live Demo

> 🔗 **[Try it live on Streamlit Cloud](#)** ← *(deploy and paste link here)*

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io)
- **LLM:** [OpenAI GPT-4o-mini](https://platform.openai.com) via `openai` Python SDK
- **ML Models:** scikit-learn (Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR / KNN / SVC)
- **Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/)
- **Data:** Pandas + NumPy

---

## 📁 Project Structure

```
ML-model-selector/
├── app.py                   # Main Streamlit app — routing & UI shell
├── modules/
│   ├── data_processor.py    # CSV loading, validation, profiling
│   ├── llm_analyst.py       # OpenAI API integration & prompt engineering
│   ├── visualizer.py        # Interactive Plotly chart builder
│   └── ml_trainer.py        # Multi-model training & cross-validation
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/Elfer09/ML-model-selector.git
cd ML-model-selector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**

Option A — via `.env` file:
```bash
cp .env.example .env
# Edit .env and add your key
```

Option B — directly in the app sidebar (no setup needed, key is not stored).

**4. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Connect your repo, set `app.py` as the entry point
4. Add `OPENAI_API_KEY` under **Settings → Secrets**
5. Deploy — you get a public URL instantly

---

## 💡 How It Works

### Chat with Your Data
The dataset is profiled on upload (shape, dtypes, describe stats, sample rows). This profile is injected into the LLM's system prompt as structured context. Every user question is answered grounded in that context — no hallucination risk on column names or statistics.

### ML Training
Models are wrapped in a `Pipeline` with `SimpleImputer` → `StandardScaler` → model. All models are evaluated with **5-fold cross-validation**, so results are robust even on small datasets. Results are ranked and visualized in an interactive bar chart.

---

## 📌 Roadmap

- [ ] Export chat history as PDF report
- [ ] LLM-suggested visualizations ("what chart should I use for this?")
- [ ] Support for Excel (.xlsx) uploads
- [ ] Feature importance explanations (SHAP)
- [ ] Deploy as Docker container

---

## 🙋 About

Built by **Eleni** as part of an AI engineering portfolio.  
Feedback, issues, and PRs are welcome.

[![GitHub](https://img.shields.io/badge/GitHub-Elfer09-black?logo=github)](https://github.com/Elfer09)
