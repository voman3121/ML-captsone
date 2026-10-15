# ML Capstone — Streamlit App

A beginner-friendly Streamlit interface for exploring traditional machine learning algorithms for regression, classification, and clustering. The original Jupyter notebooks and algorithms are retained in this repository.

## Current status

The app is **not deployed yet**. You can run it locally by following the steps below. Deployment instructions for Streamlit Community Cloud are included near the end.

## What the app includes

- **Regression:** compares ten regressors on the flight-fare training data and can evaluate an uploaded fare dataset when it has the fields needed to construct the model inputs. The five core regression models are Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and K-Nearest Neighbors; the page also includes the other implemented regression algorithms.
- **Classification:** evaluates ten classifiers on a labeled cardio dataset and reports classification metrics and visualizations. The uploaded CSV needs a target/label column and usable feature columns.
- **Clustering:** K-Means on the financial-risk data and Agglomerative clustering on Iris-style measurements, with cluster evaluation metrics and plots. An Iris CSV can be supplied for Agglomerative clustering.

The precise input columns and supported options are described on each page in the app. Bundled data files used by the app are in `data/`.

## Run locally (Windows PowerShell)

Run these commands from the repository root (the directory containing `README.md`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r notebooks/requirements-app.txt
streamlit run notebooks/app.py
```

Streamlit prints a local address, usually `http://localhost:8501`; open it in your browser. If PowerShell blocks virtual-environment activation, use a terminal where activation is permitted or run the environment's Python directly: `.\.venv\Scripts\python.exe -m streamlit run notebooks/app.py`.

For macOS/Linux, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r notebooks/requirements-app.txt
streamlit run notebooks/app.py
```

`notebooks/requirements-app.txt` installs the project dependencies plus Streamlit, Plotly, and Excel support. `notebooks/requirements.txt` provides the standard dependency filename used by Streamlit Community Cloud for this app entry point.

## Deploy with Streamlit Community Cloud

Deployment has not been performed. To publish the app:

1. Push the project to a GitHub repository that your Streamlit Community Cloud account can access.
2. Confirm the app's bundled data files (including `data/Data_Train.xlsx` and `data/credit_risk_dataset.csv`, if present in your checkout) are committed and available to the app. Keep private or sensitive data out of a public repository. Classification data can be uploaded by the person using the app.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
4. Create an app and select the repository and branch. Set the app's main file path to `notebooks/app.py`.
5. Deploy. Community Cloud installs dependencies from `notebooks/requirements.txt` and provides a hosted `*.streamlit.app` address. Check the deployment logs if a dependency or data-file path needs adjustment.

The app uses repository-relative data paths, so preserve the `data/` folder structure when pushing and deploying. See Streamlit's guides for [deploying an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) and [managing dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
