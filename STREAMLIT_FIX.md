# Streamlit Cloud installation fix

This cloud-safe requirements file intentionally omits optional packages that are not required to boot the app:

- lightgbm (the model automatically falls back to sklearn GradientBoostingRegressor)
- pyarrow (Streamlit/pandas resolve their own compatible dependency when needed)
- joblib (not imported by this project)

Recommended Python version for Streamlit Community Cloud: **3.11** or **3.12**.

If the app was originally deployed with another Python version and still fails, delete/redeploy the app and select Python 3.11 in Advanced settings before deploying.
