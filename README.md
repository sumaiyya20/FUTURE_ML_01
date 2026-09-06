# FUTURE_ML_01 — Sales & Demand Forecasting for Businesses
### Machine Learning Internship — Task 1 (Future Interns, 2026)

A business-friendly web app that predicts future sales/demand from historical
data and presents the results the way a **store owner, startup founder, or
business manager** would want to see them — not just raw model output.

🔗 **Live demo:** _add your Streamlit Cloud link here after deploying (see below)_

---

## ✅ Task Requirements Covered

| Requirement | Where it's done |
|---|---|
| Data cleaning & handling missing values | `forecasting.py → clean_data()` |
| Time-based feature engineering (date, month, seasonality) | `forecasting.py → make_features()` |
| Forecasting with regression / time-series methods | `forecasting.py → train_model()` (Random Forest or Linear Regression) |
| Model evaluation & error analysis | `forecasting.py → evaluate_model()`, error chart in `app.py` |
| Business-friendly visualization | Interactive Plotly charts in `app.py` |
| Plain-English business summary | `forecasting.py → business_summary()` |
| Deliverable: forecast + visuals + explanation | Full `app.py` Streamlit dashboard |

---

## 🧠 What It Does

1. **Loads data** — a built-in synthetic 4-year, multi-store sample dataset, or your own CSV.
2. **Cleans it** — parses dates, fixes missing values, removes negative/outlier sales.
3. **Engineers features** — day of week, month, quarter, cyclical seasonality encodings,
   lag features (1/7/14/28 days), and rolling averages.
4. **Trains a model** — Random Forest Regressor (recommended) or Linear Regression, evaluated on
   a held-out, most-recent time slice (proper time-series validation — no shuffling).
5. **Evaluates errors** — MAE, RMSE, MAPE, and an easy-to-read "Approx. Accuracy %".
6. **Forecasts the future** — recursive multi-step forecasting for a user-chosen horizon (7–90 days).
7. **Explains it in business terms** — a plain-English takeaway on expected demand direction,
   peak days, and how to act on it (inventory, staffing, cash flow).

---

## 🗂 Project Structure

```
FUTURE_ML_01/
├── app.py                     # Streamlit web app (the deliverable)
├── forecasting.py             # ML pipeline: cleaning, features, model, evaluation
├── generate_sample_data.py    # Creates the built-in synthetic dataset
├── data/
│   └── sales_data.csv         # Sample dataset (4 yrs, 3 stores, 3 categories)
├── requirements.txt
├── .streamlit/config.toml     # App theme
└── README.md
```

---

## 💻 Run Locally

```bash
git clone https://github.com/<your-username>/FUTURE_ML_01.git
cd FUTURE_ML_01
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## ☁️ Deploy for Free (GitHub + Streamlit Community Cloud)

1. **Push this project to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Sales & Demand Forecasting - ML Internship Task 1"
   git branch -M main
   git remote add origin https://github.com/<your-username>/FUTURE_ML_01.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → select your repo → branch `main` → main file path `app.py`.
4. Click **Deploy**. Your live URL will look like:
   `https://<your-username>-FUTURE_ML_01.streamlit.app`
5. Paste that link into the "Live demo" line at the top of this README, and into your
   task submission.

> **Alternative platforms:** the app also deploys as-is to **Hugging Face Spaces**
> (SDK: Streamlit) or **Render** (Web Service, start command
> `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).

---

## 📊 Using Your Own Dataset

Upload any CSV with at least:
- a **date column** (any parseable date format)
- a **sales / demand / quantity / revenue column** (numeric)

If you have multiple stores/products in one file (like the sample dataset), the app
automatically aggregates to a total daily series before forecasting. Recommended
public datasets if you want to try alternatives:

- [Store Sales – Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- [Superstore Sales Dataset (Kaggle)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- [Online Retail Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/online+retail)

---

## 🛠 Tech Stack

- **Python**, **Pandas**, **NumPy** — data handling
- **Scikit-learn** — Random Forest / Linear Regression forecasting
- **Plotly** — interactive, business-friendly visualizations
- **Streamlit** — web app framework & free deployment

---

## 📌 Skills Demonstrated

Time-series analysis · forecasting · feature engineering · model evaluation ·
business interpretation of ML output · end-to-end deployment.

---

## 📄 License

MIT — free to use and adapt for learning purposes.
