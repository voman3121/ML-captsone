# ML Capstone — Streamlit App

A beginner-friendly Streamlit application for exploring traditional machine learning in regression, classification, and clustering. The original Jupyter notebooks are kept in `notebooks/` and are not changed by the app.

## Current status

The app is deployed on Streamlit Community Cloud: [Open ML Capstone](https://ml-captsone-team-20.streamlit.app/). You can also run it locally using the commands below.

## Application features

- **Regression:** compare ten flight-fare regressors, explore model errors and fare data, evaluate a labeled upload, predict fares from an uploaded file, or enter one editable flight record for a fare estimate. The bundled AeroFare training file is `data/Data_Train.xlsx`.
- **Classification:** upload a labeled cardiovascular dataset with a binary `cardio` target, compare ten classifiers, review the evaluation metrics and plots, then enter one editable patient record for a class prediction. The uploaded labeled dataset supplies the model training data.
- **Clustering:** run K-Means and Agglomerative clustering on the same cleaned and scaled FinRisk borrower features from `clustering_merged.ipynb`. Compare Silhouette, Davies–Bouldin, and Calinski–Harabasz scores on a shared sample. You can also enter one borrower record and see the K-Means assignment alongside Agglomerative's nearest-cluster-profile estimate. `loan_status` is excluded from clustering features.

The app uses `data/credit_risk_dataset.csv` for FinRisk when that file is available, or you can upload the CSV in the app. The clustering page samples large inputs for responsiveness; Agglomerative comparison uses up to 2,500 borrowers because its memory cost grows rapidly with sample size.

## Run locally

Run from the repository root (the directory containing this README).

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r notebooks/requirements-app.txt
streamlit run notebooks/app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. If PowerShell blocks environment activation, run Streamlit directly through the environment's Python:

```powershell
.\.venv\Scripts\python.exe -m streamlit run notebooks/app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r notebooks/requirements-app.txt
streamlit run notebooks/app.py
```

`notebooks/requirements-app.txt` is the local install list. `notebooks/requirements.txt` is the standard dependency file Community Cloud detects next to `notebooks/app.py`; both list the packages required by the app.

## Deploy with Streamlit Community Cloud

1. Push the latest project to a GitHub repository that your Streamlit Community Cloud account can access.
2. Make sure the repository includes `notebooks/app.py`, `notebooks/requirements.txt`, and the needed data files in the root `data/` folder, including `Data_Train.xlsx` and `credit_risk_dataset.csv` if the app should load them automatically. Keep the directory structure intact.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
4. Select **Create app**, choose the repository and branch, and set the main file path to `notebooks/app.py`.
5. Deploy. Community Cloud installs the app dependencies and gives you a hosted `*.streamlit.app` URL. Review the deployment logs if a dependency or data path needs attention.

See Streamlit's guides for [deploying an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency files](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), and [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).


