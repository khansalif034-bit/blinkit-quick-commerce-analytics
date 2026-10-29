# Blinkit Quick Commerce Sales and Predictive Analytics

Interactive Streamlit dashboard (Python, Pandas, Plotly) built on a consolidated Blinkit master dataset.

## Pages
1. Executive Overview: KPIs, sales trend (day/week/month), category revenue, payment mix, weekday pattern
2. Customer Intelligence: segments, repeat rate, spend vs frequency, top customers
3. Predictive Intelligence: repeat-purchase propensity, probability bands, high-value at-risk roster
4. Delivery & Retention: SLA delay rate, root causes, delay vs repeat behaviour
5. Data Explorer: data-quality report, column picker, CSV export

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py` and the CSV (any file name containing `master`) in the same folder.
