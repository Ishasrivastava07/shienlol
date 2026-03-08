# Shein APAC Intelligence Suite

A Streamlit dashboard covering 3 APAC problem statements:
- **PS1** Retention & Churn — RFM, churn prediction, uplift segmentation
- **PS2** Logistics & Margin — warehouse hub analysis, delivery issues, gross margin
- **PS3** Sustainable Growth — campaign ROI, discount dependency, CLV measurement

## Files
- `app.py` — main dashboard
- `shein_apac_synthetic.csv` — 200-row synthetic dataset
- `requirements.txt` — dependencies
- `.streamlit/config.toml` — dark pink theme

## Deploy
```bash
pip install -r requirements.txt
streamlit run app.py
```
Or push to GitHub and deploy via [share.streamlit.io](https://share.streamlit.io).

## Dependencies
Only `streamlit`, `pandas`, `numpy`. ML models (Decision Tree, Random Forest, Gradient Boosting)
are implemented from scratch — no scikit-learn, no plotly, no scipy.
