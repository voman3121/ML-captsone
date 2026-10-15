from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = go = None
    PLOTLY_AVAILABLE = False


st.set_page_config(page_title="ML Studio | Capstone", page_icon="🧪", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_FILE = PROJECT_ROOT / "data" / "Data_Train.xlsx"
RAW_FEATURES = [
    "Airline", "Date_of_Journey", "Source", "Destination", "Dep_Time",
    "Duration", "Total_Stops", "Additional_Info",
]
NUMERIC_FEATURES = [
    "journey_day", "journey_month", "dep_hour", "dep_minute", "duration_minutes", "total_stops_n",
]
CATEGORICAL_FEATURES = ["Airline", "Source", "Destination", "additional_info_grouped"]
MODEL_NAMES = ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet",
               "Polynomial Regression", "Decision Tree Regressor", "Random Forest Regressor",
               "Gradient Boosting Regressor", "Support Vector Regressor", "K-Nearest Neighbors Regressor"]
UPLOAD_EVALUATION_MAX_ROWS = 5_000

COLUMN_ALIASES = {
    "Airline": ["airline", "carrier", "airline_name"],
    "Date_of_Journey": ["date_of_journey", "journey_date", "flight_date", "travel_date", "date_of_travel"],
    "Source": ["source", "origin", "from", "departure_city", "source_city"],
    "Destination": ["destination", "dest", "to", "arrival_city", "destination_city"],
    "Dep_Time": ["dep_time", "departure_time", "depart_time", "departure"],
    "Duration": ["duration", "flight_duration", "journey_duration"],
    "Total_Stops": ["total_stops", "stops", "number_of_stops", "stop_count"],
    "Additional_Info": ["additional_info", "additional_information", "info"],
    "journey_day": ["journey_day", "flight_day", "day_of_journey"],
    "journey_month": ["journey_month", "flight_month", "month_of_journey"],
    "dep_hour": ["dep_hour", "departure_hour"],
    "dep_minute": ["dep_minute", "departure_minute"],
    "duration_minutes": ["duration_minutes", "duration_min", "flight_duration_minutes"],
    "total_stops_n": ["total_stops_n", "stops_numeric", "stop_count_numeric"],
}


def normalize_flight_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common alternate flight column names to the names used by the notebook."""
    df = frame.copy()
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_"): col
        for col in df.columns
    }
    rename = {}
    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in normalized:
                rename[normalized[alias]] = target
                break
    return df.rename(columns=rename)


def parse_duration(value: object) -> int:
    text = str(value)
    hours = re.search(r"(\d+)h", text)
    minutes = re.search(r"(\d+)m", text)
    return (int(hours.group(1)) if hours else 0) * 60 + (int(minutes.group(1)) if minutes else 0)


def prepare_flights(frame: pd.DataFrame, rare_categories: set[str] | None = None) -> pd.DataFrame:
    df = normalize_flight_columns(frame)
    # These columns are not used by the trained models, so test files may omit them.
    for col in ["Airline", "Source", "Destination", "Additional_Info"]:
        if col not in df.columns:
            df[col] = "Unknown"

    # Accept either raw notebook fields or already engineered values from another dataset.
    if "Date_of_Journey" not in df.columns and not {"journey_day", "journey_month"}.issubset(df.columns):
        missing_date = True
    else:
        missing_date = False
    if "Dep_Time" not in df.columns and not {"dep_hour", "dep_minute"}.issubset(df.columns):
        missing_departure = True
    else:
        missing_departure = False
    if "Duration" not in df.columns and "duration_minutes" not in df.columns:
        missing_duration = True
    else:
        missing_duration = False
    if "Total_Stops" not in df.columns and "total_stops_n" not in df.columns:
        missing_stops = True
    else:
        missing_stops = False
    missing = [label for absent, label in [
        (missing_date, "journey date (or journey_day and journey_month)"),
        (missing_departure, "departure time (or dep_hour and dep_minute)"),
        (missing_duration, "duration (or duration_minutes)"),
        (missing_stops, "number of stops (or total_stops_n)"),
    ] if absent]
    if missing:
        available = ", ".join(map(str, frame.columns)) or "none"
        raise ValueError("This file is missing model inputs: " + "; ".join(missing)
                         + f". Columns found: {available}. See the accepted column names below.")
    if "Date_of_Journey" in df.columns:
        journey_date = pd.to_datetime(df["Date_of_Journey"], format="mixed", dayfirst=True, errors="coerce")
        df["journey_day"], df["journey_month"] = journey_date.dt.day, journey_date.dt.month
    if "Dep_Time" in df.columns:
        dep = pd.to_datetime(df["Dep_Time"].astype(str), format="%H:%M", errors="coerce")
        df["dep_hour"], df["dep_minute"] = dep.dt.hour, dep.dt.minute
    if "Duration" in df.columns:
        duration_values = df["Duration"]
        if pd.api.types.is_numeric_dtype(duration_values):
            df["duration_minutes"] = pd.to_numeric(duration_values, errors="coerce")
        else:
            df["duration_minutes"] = duration_values.map(parse_duration)
    stop_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    if "Total_Stops" in df.columns:
        stop_text = df["Total_Stops"].astype(str).str.strip().str.lower()
        mapped_stops = stop_text.map(stop_map)
        df["total_stops_n"] = mapped_stops.fillna(pd.to_numeric(stop_text, errors="coerce"))
    df["additional_info_grouped"] = df["Additional_Info"].fillna("Unknown").astype(str)
    if rare_categories is not None:
        df.loc[~df["additional_info_grouped"].isin(rare_categories), "additional_info_grouped"] = "Other"
    df["Airline"] = df["Airline"].fillna("Unknown").astype(str)
    for col in ["Source", "Destination"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


@st.cache_resource(show_spinner="Preparing the ten regression models from AeroFare…")
def load_regression_models():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_FILE}")
    raw = pd.read_excel(TRAINING_FILE)
    df = raw.dropna(subset=["Route", "Total_Stops"]).drop_duplicates().copy()
    q1, q3 = df["Price"].quantile([0.25, 0.75])
    fence = 3 * (q3 - q1)
    df = df[df["Price"].between(q1 - fence, q3 + fence)].copy()
    if len(df) > 5_000:
        df = df.sample(5_000, random_state=42)
    info_freq = df["Additional_Info"].fillna("Unknown").astype(str).value_counts(normalize=True)
    frequent_info = set(info_freq[info_freq >= 0.01].index)
    x = prepare_flights(df, frequent_info)
    y = df["Price"].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def make_preprocessor():
        return make_pipeline(ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL_FEATURES),
             ("num", "passthrough", NUMERIC_FEATURES)],
            remainder="drop",
        ), StandardScaler())

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=5000, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                                  max_depth=3, random_state=42),
        "Support Vector Regressor": SVR(C=10, kernel="rbf"),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=7),
    }
    fitted, scores, validation_predictions = {}, {}, {}
    preprocessor = make_preprocessor()
    train_matrix = preprocessor.fit_transform(x_train)
    test_matrix = preprocessor.transform(x_test)
    for name, estimator in models.items():
        estimator.fit(train_matrix, y_train)
        pred = estimator.predict(test_matrix)
        validation_predictions[name] = pred
        fitted[name] = (estimator, preprocessor, None)
        scores[name] = {"R²": r2_score(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                        "MAE": mean_absolute_error(y_test, pred)}

    # Polynomial features are applied to the six numeric inputs only, as in the notebook.
    poly_pre = make_preprocessor()
    poly_train, poly_test = poly_pre.fit_transform(x_train), poly_pre.transform(x_test)
    transformer = poly_pre.named_steps["columntransformer"]
    cat_count = len(transformer.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    # The ColumnTransformer places categories before numeric inputs.
    numeric_start = cat_count
    degree = 2
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_train_num = poly.fit_transform(poly_train[:, numeric_start:])
    poly_test_num = poly.transform(poly_test[:, numeric_start:])
    poly_train_full = np.column_stack([poly_train_num, poly_train[:, :numeric_start]])
    poly_test_full = np.column_stack([poly_test_num, poly_test[:, :numeric_start]])
    poly_model = LinearRegression().fit(poly_train_full, y_train)
    poly_pred = poly_model.predict(poly_test_full)
    validation_predictions["Polynomial Regression"] = poly_pred
    scores["Polynomial Regression"] = {"R²": r2_score(y_test, poly_pred),
                                       "RMSE": mean_squared_error(y_test, poly_pred) ** 0.5,
                                       "MAE": mean_absolute_error(y_test, poly_pred)}
    fitted["Polynomial Regression"] = (poly_model, poly_pre, (poly, numeric_start))

    # Refit selected models on all labelled records so they can predict a separate uploaded file.
    all_pre = make_preprocessor()
    all_matrix = all_pre.fit_transform(x)
    for name, estimator in models.items():
        estimator.fit(all_matrix, y)
        fitted[name] = (estimator, all_pre, None)
    all_poly_pre = make_preprocessor()
    all_matrix = all_poly_pre.fit_transform(x)
    all_transformer = all_poly_pre.named_steps["columntransformer"]
    all_cat_count = len(all_transformer.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_poly = PolynomialFeatures(degree=2, include_bias=False)
    all_num = all_poly.fit_transform(all_matrix[:, all_cat_count:])
    all_full = np.column_stack([all_num, all_matrix[:, :all_cat_count]])
    fitted["Polynomial Regression"] = (LinearRegression().fit(all_full, y), all_poly_pre, (all_poly, all_cat_count))
    return fitted, scores, frequent_info, len(df), y_test.to_numpy(), validation_predictions


def predict_rows(bundle, rows: pd.DataFrame, frequent_info: set[str]) -> np.ndarray:
    estimator, preprocessor, poly_bundle = bundle
    features = prepare_flights(rows, frequent_info)
    matrix = preprocessor.transform(features)
    if poly_bundle is not None:
        poly, numeric_start = poly_bundle
        expanded = poly.transform(matrix[:, numeric_start:])
        matrix = np.column_stack([expanded, matrix[:, :numeric_start]])
    return estimator.predict(matrix)


def evaluate_uploaded_fare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, int, np.ndarray, dict[str, np.ndarray]]:
    """Train and compare the five models when the upload itself contains fare labels."""
    df = frame.copy()
    target_col = next((c for c in df.columns if str(c).strip().lower() in {"fare", "price"}), None)
    if target_col is None:
        raise ValueError("No fare label found. To evaluate this dataset, include a `fare` or `price` column.")
    y = pd.to_numeric(df[target_col], errors="coerce")
    # Exclude other fare columns to prevent the target being inferred from related fare labels.
    fare_cols = [c for c in df.columns if "fare" in str(c).lower() or str(c).lower() == "price"]
    excluded = set(fare_cols)
    for col in df.columns:
        key = str(col).strip().lower()
        if key == "tbl" or key.endswith("_id") or key.startswith("geocoded") or "apk" in key:
            excluded.add(col)
    x = df.drop(columns=list(excluded), errors="ignore").copy()
    valid = y.notna()
    x, y = x.loc[valid].copy(), y.loc[valid]
    # Remove columns that carry no usable values, then impute remaining gaps.
    x = x.dropna(axis=1, how="all")
    if len(x) < 10:
        raise ValueError("At least 10 rows with numeric fare labels are needed for a train/validation split.")
    if x.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding fare and identifier columns.")
    # Bound run time and memory for large public airline datasets; sampling is reproducible.
    if len(x) > UPLOAD_EVALUATION_MAX_ROWS:
        x = x.sample(n=UPLOAD_EVALUATION_MAX_ROWS, random_state=42)
        y = y.loc[x.index]
    numeric = list(x.select_dtypes(include=np.number).columns)
    categorical = [c for c in x.columns if c not in numeric]
    transformers = []
    if categorical:
        transformers.append(("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10, max_categories=30)), categorical))
    if numeric:
        transformers.append(("num", make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()), numeric))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
    train_matrix = preprocessor.fit_transform(x_train)
    valid_matrix = preprocessor.transform(x_valid)
    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                                  max_depth=3, random_state=42),
        "Support Vector Regressor": SVR(C=10, kernel="rbf"),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=7),
    }
    results, validation_predictions = {}, {}
    for name, model in estimators.items():
        model.fit(train_matrix, y_train)
        prediction = model.predict(valid_matrix)
        validation_predictions[name] = prediction
        results[name] = {"R²": r2_score(y_valid, prediction),
                         "RMSE": mean_squared_error(y_valid, prediction) ** 0.5,
                         "MAE": mean_absolute_error(y_valid, prediction)}

    # Add quadratic terms for numeric inputs only, preserving one-hot columns as-is.
    cat_width = 0
    if categorical:
        cat_width = len(preprocessor.named_transformers_["cat"].named_steps["onehotencoder"].get_feature_names_out(categorical))
    if numeric:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_train_num = poly.fit_transform(train_matrix[:, cat_width:])
        poly_valid_num = poly.transform(valid_matrix[:, cat_width:])
        poly_train = np.column_stack([poly_train_num, train_matrix[:, :cat_width]])
        poly_valid = np.column_stack([poly_valid_num, valid_matrix[:, :cat_width]])
    else:
        poly_train, poly_valid = train_matrix, valid_matrix
    model = LinearRegression().fit(poly_train, y_train)
    prediction = model.predict(poly_valid)
    validation_predictions["Polynomial Regression"] = prediction
    results["Polynomial Regression"] = {"R²": r2_score(y_valid, prediction),
                                        "RMSE": mean_squared_error(y_valid, prediction) ** 0.5,
                                        "MAE": mean_absolute_error(y_valid, prediction)}
    return (pd.DataFrame(results).T.rename_axis("Algorithm"), str(target_col), len(x),
            y_valid.to_numpy(), validation_predictions)


def show_metric_chart(scores: pd.DataFrame, key: str) -> None:
    metric = st.selectbox("Metric to compare", ["R²", "RMSE", "MAE"], key=key)
    if not PLOTLY_AVAILABLE:
        st.bar_chart(scores[metric])
        return
    fig = px.bar(scores.reset_index(), x="Algorithm", y=metric, color="Algorithm", text_auto=".3f",
                 title=f"Model comparison · {metric}", color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=metric, height=380)
    st.plotly_chart(fig, use_container_width=True)


def show_animated_prediction_chart(actual: np.ndarray, predicted: np.ndarray, model_name: str, key: str) -> None:
    # Sort by the actual fare so the animation progresses from low to high fares.
    order = np.argsort(actual)
    actual, predicted = np.asarray(actual)[order], np.asarray(predicted)[order]
    if len(actual) > 1000:
        sample = np.linspace(0, len(actual) - 1, 1000).astype(int)
        actual, predicted = actual[sample], predicted[sample]
    if not PLOTLY_AVAILABLE:
        st.info("Install Plotly to enable the animated chart: `python -m pip install plotly`")
        st.scatter_chart(pd.DataFrame({"Actual fare (₹)": actual, "Predicted fare (₹)": predicted}),
                         x="Actual fare (₹)", y="Predicted fare (₹)")
        return
    frame_count = min(12, len(actual))
    step = max(1, int(np.ceil(len(actual) / frame_count)))
    frames = []
    for end in range(step, len(actual) + step, step):
        count = min(end, len(actual))
        frames.append(go.Frame(name=str(count), data=[go.Scatter(
            x=actual[:count], y=predicted[:count], mode="markers",
            marker={"color": "#0F766E", "size": 7, "opacity": 0.65},
            hovertemplate="Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>")], traces=[0]))
    max_value = float(max(np.max(actual), np.max(predicted)))
    min_value = float(min(np.min(actual), np.min(predicted)))
    fig = go.Figure(
        data=[go.Scatter(x=actual[:1], y=predicted[:1], mode="markers", name="Validation rows",
                         marker={"color": "#0F766E", "size": 7, "opacity": 0.65}),
              go.Scatter(x=[min_value, max_value], y=[min_value, max_value], mode="lines",
                         name="Perfect estimate", line={"color": "#D97706", "dash": "dash"})],
        frames=frames,
    )
    fig.update_layout(
        title=f"Actual vs predicted fares · {model_name}", xaxis_title="Actual fare (₹)",
        yaxis_title="Predicted fare (₹)", height=520,
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.02, "y": 1.12,
                      "buttons": [{"label": "▶ Play", "method": "animate",
                                   "args": [None, {"frame": {"duration": 300, "redraw": True},
                                             "transition": {"duration": 150}, "fromcurrent": True}]},
                                  {"label": "↺ Reset", "method": "animate",
                                   "args": [[frames[0].name], {"frame": {"duration": 0, "redraw": True},
                                                               "transition": {"duration": 0}}]}]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Rows revealed: "},
                  "steps": [{"label": frame.name, "method": "animate",
                             "args": [[frame.name], {"frame": {"duration": 0, "redraw": True},
                                                       "transition": {"duration": 0}}]} for frame in frames]}],
    )
    st.caption("Use Play or the slider to reveal validation records from lower to higher actual fares. Points near the dashed line have smaller prediction errors.")
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_uploaded_data_charts(frame: pd.DataFrame, target_col: str, key: str) -> None:
    fare = pd.to_numeric(frame[target_col], errors="coerce").dropna()
    if len(fare):
        q1, q3 = fare.quantile([0.25, 0.75])
        st.caption(f"Target summary · n={len(fare):,} · mean={fare.mean():,.2f} · median={fare.median():,.2f} · IQR={q1:,.2f}–{q3:,.2f}")
    if not PLOTLY_AVAILABLE:
        st.info("Install Plotly to enable animated visualizations: `python -m pip install plotly`")
        st.bar_chart(fare.value_counts(bins=40).sort_index())
        lower = {str(c).strip().lower(): c for c in frame.columns}
        miles_col = lower.get("nsmiles")
        if miles_col:
            points = pd.DataFrame({"Distance (miles)": pd.to_numeric(frame[miles_col], errors="coerce"),
                                   "Fare": pd.to_numeric(frame[target_col], errors="coerce")}).dropna()
            st.scatter_chart(points, x="Distance (miles)", y="Fare")
        return
    hist = px.histogram(x=fare, nbins=40, labels={"x": "Fare", "y": "Rows"},
                        title="Fare distribution in the uploaded dataset", color_discrete_sequence=["#0F766E"])
    hist.update_layout(height=360, showlegend=False)
    st.plotly_chart(hist, use_container_width=True, key=f"{key}-hist")

    numeric = frame.select_dtypes(include=np.number).copy()
    numeric[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Numeric feature correlation")
        st.plotly_chart(heat, use_container_width=True, key=f"{key}-corr")

    categorical = [c for c in frame.select_dtypes(exclude=np.number).columns if c != target_col]
    if categorical:
        cat_col = categorical[0]
        summary = pd.DataFrame({cat_col: frame[cat_col].astype(str), "Fare": pd.to_numeric(frame[target_col], errors="coerce")})
        summary = summary.groupby(cat_col, dropna=False)["Fare"].agg(["mean", "count"]).query("count >= 5")
        summary = summary.sort_values("count", ascending=False).head(15).reset_index()
        if len(summary):
            cat_fig = px.bar(summary, x=cat_col, y="mean", color="count", title=f"Average fare by {cat_col} (common groups)",
                             labels={"mean": "Average fare", "count": "Rows"}, color_continuous_scale="Teal")
            st.plotly_chart(cat_fig, use_container_width=True, key=f"{key}-category")

    lower = {str(c).strip().lower(): c for c in frame.columns}
    miles_col = lower.get("nsmiles")
    year_col = next((lower[k] for k in ["year", "yr"] if k in lower), None)
    if miles_col is not None:
        chart_data = frame[[miles_col, target_col] + ([year_col] if year_col else [])].copy()
        chart_data[miles_col] = pd.to_numeric(chart_data[miles_col], errors="coerce")
        chart_data[target_col] = pd.to_numeric(chart_data[target_col], errors="coerce")
        chart_data = chart_data.dropna(subset=[miles_col, target_col])
        if len(chart_data) > 5000:
            chart_data = chart_data.sample(5000, random_state=42)
        if year_col:
            chart_data[year_col] = chart_data[year_col].astype(str)
            scatter = px.scatter(chart_data, x=miles_col, y=target_col, animation_frame=year_col,
                                 title="How fare and distance vary by year",
                                 labels={miles_col: "Distance (miles)", target_col: "Fare"},
                                 color_discrete_sequence=["#2563EB"], opacity=0.65)
            scatter.update_layout(height=480)
            st.caption("Use the animation controls to move through years. This shows association, not causation.")
            st.plotly_chart(scatter, use_container_width=True, key=f"{key}-distance-year")
        else:
            scatter = px.scatter(chart_data, x=miles_col, y=target_col,
                                 title="Fare vs flight distance",
                                 labels={miles_col: "Distance (miles)", target_col: "Fare"},
                                 color_discrete_sequence=["#2563EB"], opacity=0.65)
            scatter.update_layout(height=460)
            st.plotly_chart(scatter, use_container_width=True, key=f"{key}-distance")


def show_regression_diagnostics(actual: np.ndarray, predicted: np.ndarray, model_name: str,
                                key: str, bundle: tuple | None = None) -> None:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    residuals = actual - predicted
    st.subheader(f"Prediction diagnostics · {model_name}")
    if PLOTLY_AVAILABLE:
        sample = np.linspace(0, len(actual) - 1, min(2500, len(actual))).astype(int)
        scatter = px.scatter(x=actual[sample], y=predicted[sample], color=np.abs(residuals[sample]),
                             color_continuous_scale="Viridis", labels={"x": "Actual fare", "y": "Predicted fare",
                                                                      "color": "Absolute error"},
                             title="Predicted vs actual fares")
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        scatter.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                          line={"color": "#D97706", "dash": "dash"})
        scatter.update_layout(height=430)
        residual_hist = px.histogram(x=residuals, nbins=40, labels={"x": "Residual (actual − predicted)", "y": "Rows"},
                                     title="Residual distribution", color_discrete_sequence=["#0F766E"])
        residual_hist.add_vline(x=0, line_dash="dash", line_color="#D97706")
        error_scatter = px.scatter(x=predicted[sample], y=residuals[sample],
                                   labels={"x": "Predicted fare", "y": "Residual"},
                                   title="Residuals vs predicted fare", color_discrete_sequence=["#2563EB"])
        error_scatter.add_hline(y=0, line_dash="dash", line_color="#D97706")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(scatter, use_container_width=True, key=f"{key}-actual-predicted")
        with c2:
            st.plotly_chart(residual_hist, use_container_width=True, key=f"{key}-residual-hist")
        st.plotly_chart(error_scatter, use_container_width=True, key=f"{key}-residual-vs-predicted")
    else:
        st.scatter_chart(pd.DataFrame({"Actual fare": actual, "Predicted fare": predicted}),
                         x="Actual fare", y="Predicted fare")
        st.bar_chart(pd.Series(residuals).value_counts(bins=30).sort_index())

    if bundle is not None:
        estimator, preprocessor, _ = bundle
        importance = getattr(estimator, "feature_importances_", None)
        if importance is None and hasattr(estimator, "coef_"):
            importance = np.abs(np.ravel(estimator.coef_))
        if importance is not None:
            try:
                names = preprocessor.named_steps["columntransformer"].get_feature_names_out()
            except Exception:
                names = np.array([f"Feature {i + 1}" for i in range(len(importance))])
            if len(names) == len(importance):
                top = pd.Series(importance, index=names).sort_values(ascending=False).head(15).sort_values()
                if PLOTLY_AVAILABLE:
                    fig = px.bar(x=top.values, y=top.index, orientation="h",
                                 labels={"x": "Importance / absolute coefficient", "y": "Feature"},
                                 title=f"Top feature signals · {model_name}", color=top.values,
                                 color_continuous_scale="Teal")
                    st.plotly_chart(fig, use_container_width=True, key=f"{key}-importance")
                else:
                    st.bar_chart(top)


def regression_page():
    st.title("Flight fare prediction")
    st.write("Compare all ten regression algorithms and upload flight data for fare evaluation or prediction.")
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed, so charts are shown without animation. Install app dependencies with `pip install -r notebooks/requirements-app.txt` and restart Streamlit for animated charts.")
    try:
        fitted, scores, frequent_info, row_count, validation_actual, validation_predictions = load_regression_models()
    except Exception as exc:
        st.error(f"Could not prepare the regression models: {exc}")
        st.info("Keep the labelled AeroFare file at `data/Data_Train.xlsx` in the project root.")
        return

    st.caption(f"Training source: AeroFare · {row_count:,} sampled, cleaned labelled records · predictions are estimates")
    left, right = st.columns([1, 2])
    with left:
        selected = st.selectbox("Regression algorithm", MODEL_NAMES)
        st.metric("Validation R²", f"{scores[selected]['R²']:.3f}")
        st.metric("Validation MAE", f"₹{scores[selected]['MAE']:,.0f}")
    with right:
        st.subheader("Compare all ten algorithms")
        comparison = pd.DataFrame(scores).T.rename_axis("Algorithm")
        st.dataframe(comparison.style.format({"R²": "{:.3f}", "RMSE": "₹{:,.0f}", "MAE": "₹{:,.0f}"}),
                     use_container_width=True)
    show_metric_chart(pd.DataFrame(scores).T.rename_axis("Algorithm"), "aerofare-metric")
    show_regression_diagnostics(validation_actual, validation_predictions[selected], selected,
                                "aerofare-diagnostics", fitted[selected])
    with st.expander("Exploratory data analysis · AeroFare"):
        raw_eda = pd.read_excel(TRAINING_FILE)
        st.write(f"Raw dataset: {raw_eda.shape[0]:,} rows × {raw_eda.shape[1]} columns")
        st.dataframe(pd.DataFrame({"Missing": raw_eda.isna().sum(), "Unique": raw_eda.nunique(),
                                   "Type": raw_eda.dtypes.astype(str)}), use_container_width=True)
        show_uploaded_data_charts(raw_eda, "Price", "aerofare-eda")
        engineered = prepare_flights(raw_eda, frequent_info)
        engineered["Price"] = pd.to_numeric(raw_eda["Price"], errors="coerce").to_numpy()
        if PLOTLY_AVAILABLE:
            corr = px.imshow(engineered[NUMERIC_FEATURES + ["Price"]].corr(), text_auto=".2f",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Engineered flight features and fare correlation")
            st.plotly_chart(corr, use_container_width=True, key="aerofare-engineered-corr")
            plot_cols = st.columns(2)
            with plot_cols[0]:
                fig = px.scatter(engineered.sample(min(4000, len(engineered)), random_state=42),
                                 x="duration_minutes", y="Price", color="total_stops_n", opacity=0.5,
                                 title="Fare vs journey duration")
                st.plotly_chart(fig, use_container_width=True, key="aerofare-duration-fare")
            with plot_cols[1]:
                fig = px.box(engineered, x="total_stops_n", y="Price", color="total_stops_n",
                             title="Fare distribution by number of stops")
                st.plotly_chart(fig, use_container_width=True, key="aerofare-stops-fare")
    st.divider()
    st.subheader("Predict fares from a flight dataset")
    st.markdown("Upload flight records as CSV or Excel. A labeled file with `fare` or `price` compares all ten algorithms. An unlabeled file with AeroFare-style flight details gets predictions from the selected model.")
    uploaded = st.file_uploader("Flight input file", type=["csv", "xlsx"], key="flight-upload")
    with st.expander("Required inputs and accepted column names"):
        st.markdown("The model needs a journey date, departure time, duration, and number of stops. You can provide raw fields or the engineered numeric equivalents shown here:")
        st.code("Journey date: Date_of_Journey / journey_date / flight_date\nDeparture: Dep_Time / departure_time (or dep_hour + dep_minute)\nDuration: Duration / flight_duration (or duration_minutes)\nStops: Total_Stops / stops / number_of_stops (or total_stops_n)\nOptional: Airline, Source (or origin), Destination (or dest)")
    if uploaded is not None:
        try:
            rows = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            if rows.empty:
                st.warning("The uploaded file has no rows.")
            else:
                st.write(f"Loaded **{len(rows):,} rows**.")
                st.dataframe(rows.head(5), use_container_width=True)
                uploaded_target = next((c for c in rows.columns if str(c).strip().lower() in {"fare", "price"}), None)
                if uploaded_target is not None:
                    st.info(f"This file contains `{uploaded_target}` labels. The app will compare all ten algorithms on a held-out validation split. Large files use a reproducible sample of up to {UPLOAD_EVALUATION_MAX_ROWS:,} rows so the evaluation can finish in the browser app.")
                    if st.button("Evaluate uploaded fare dataset", type="primary"):
                        try:
                            with st.spinner("Training and comparing the ten regression algorithms…"):
                                results, target_name, used_rows, actual, predictions_by_model = evaluate_uploaded_fare_dataset(rows)
                            st.success(f"Evaluation complete: {used_rows:,} rows, target `{target_name}`.")
                            st.dataframe(results.style.format({"R²": "{:.3f}", "RMSE": "{:,.2f}", "MAE": "{:,.2f}"}),
                                         use_container_width=True)
                            st.subheader("Explore the evaluation")
                            show_metric_chart(results, "uploaded-fare-metric")
                            show_animated_prediction_chart(actual, predictions_by_model[selected], selected,
                                                           "uploaded-fare-prediction-animation")
                            show_regression_diagnostics(actual, predictions_by_model[selected], selected,
                                                        "uploaded-fare-diagnostics")
                            st.subheader("Explore the uploaded fare data")
                            show_uploaded_data_charts(rows, target_name, "uploaded-fare-data")
                        except Exception as exc:
                            st.error(f"The evaluation could not be completed: {exc}")
                elif st.button("Predict fares", type="primary"):
                    predictions = predict_rows(fitted[selected], rows, frequent_info)
                    result = rows.copy()
                    result["Predicted_Price"] = np.round(predictions, 2)
                    st.success(f"Predicted fares for {len(result):,} flights using {selected}.")
                    st.dataframe(result, use_container_width=True)
                    st.download_button("Download predictions as CSV", result.to_csv(index=False).encode("utf-8"),
                                       "flight_fare_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read or predict this file: {exc}")


CARDIO_NUMERIC = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure"]
CARDIO_CATEGORICAL = ["cholesterol", "gluc"]
CARDIO_BINARY = ["gender", "smoke", "alco", "active"]
CARDIO_MODELS = ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes (Gaussian)",
                 "Decision Tree", "SVM (SVC)", "Random Forest", "AdaBoost", "Gradient Boosting",
                 "Bagging (Decision Tree)", "MLP (Neural Network)"]


def prepare_cardio_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = frame.copy()
    aliases = {
        "age": ["age", "age_in_days"], "age_years": ["age_years", "age_in_years"],
        "gender": ["gender", "sex"], "height": ["height", "height_cm"],
        "weight": ["weight", "weight_kg"], "ap_hi": ["ap_hi", "systolic_bp", "systolic_blood_pressure"],
        "ap_lo": ["ap_lo", "diastolic_bp", "diastolic_blood_pressure"],
        "cholesterol": ["cholesterol", "cholesterol_level"], "gluc": ["gluc", "glucose", "glucose_level"],
        "smoke": ["smoke", "smoking"], "alco": ["alco", "alcohol", "alcohol_use"],
        "active": ["active", "physically_active"], "cardio": ["cardio", "target", "heart_disease"],
    }
    normalized = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in df.columns}
    rename = {}
    for target, names in aliases.items():
        if target in df.columns:
            continue
        for alias in names:
            if alias in normalized:
                rename[normalized[alias]] = target
                break
    df = df.rename(columns=rename)
    required = ["gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    missing = [c for c in required if c not in df.columns]
    if "age" not in df.columns and "age_years" not in df.columns:
        missing.append("age or age_years")
    if missing:
        raise ValueError("Missing CardioSense fields: " + ", ".join(missing))
    if "age_years" not in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        # The notebook's CardioSense source stores age in days; accept years too.
        df["age_years"] = age / 365.25 if age.median() > 120 else age
    y = pd.to_numeric(df["cardio"], errors="coerce")
    for col in ["height", "weight", "ap_hi", "ap_lo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = y.isin([0, 1]) & df["height"].between(120, 210) & df["weight"].between(30, 200)
    valid &= df["ap_hi"].between(70, 250) & df["ap_lo"].between(40, 200) & (df["ap_hi"] > df["ap_lo"])
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    if len(df) < 20 or y.nunique() != 2:
        raise ValueError("At least 20 valid records with both cardio classes (0 and 1) are required.")
    x = df.copy()
    x["bmi"] = x["weight"] / ((x["height"] / 100) ** 2)
    x["pulse_pressure"] = x["ap_hi"] - x["ap_lo"]
    for col in CARDIO_NUMERIC + CARDIO_CATEGORICAL + CARDIO_BINARY:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x[CARDIO_NUMERIC + CARDIO_CATEGORICAL + CARDIO_BINARY], y


def evaluate_cardio_models(frame: pd.DataFrame) -> dict:
    x, y = prepare_cardio_frame(frame)
    if len(x) > 5_000:
        x, _, y, _ = train_test_split(x, y, train_size=5_000, stratify=y, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    preprocessor = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), CARDIO_NUMERIC),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CARDIO_CATEGORICAL),
        ("bin", SimpleImputer(strategy="most_frequent"), CARDIO_BINARY),
    ])
    train = preprocessor.fit_transform(x_train)
    valid = preprocessor.transform(x_valid)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "SVM (SVC)": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                        max_depth=3, random_state=42),
        "Bagging (Decision Tree)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                                                     n_estimators=100, random_state=42, n_jobs=-1),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300,
                                               early_stopping=True, random_state=42),
    }
    metrics, probabilities = {}, {}
    for name, model in models.items():
        model.fit(train, y_train)
        proba = model.predict_proba(valid)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probabilities[name] = proba
        metrics[name] = {
            "Accuracy": accuracy_score(y_valid, pred),
            "Precision": precision_score(y_valid, pred, zero_division=0),
            "Recall": recall_score(y_valid, pred, zero_division=0),
            "F1 (weighted)": f1_score(y_valid, pred, average="weighted"),
            "ROC-AUC": roc_auc_score(y_valid, proba),
        }
    return {"metrics": pd.DataFrame(metrics).T.rename_axis("Algorithm"),
            "actual": y_valid.to_numpy(), "probabilities": probabilities,
            "valid_rows": len(y_valid), "total_rows": len(x)}


def show_confusion_matrix(actual: np.ndarray, predicted: np.ndarray, model_name: str, key: str) -> None:
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    if PLOTLY_AVAILABLE:
        fig = px.imshow(matrix, text_auto=True, x=["No disease", "Disease"], y=["No disease", "Disease"],
                        color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual", "color": "Records"},
                        title=f"Confusion matrix · {model_name}")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True, key=key)
    else:
        st.subheader(f"Confusion matrix · {model_name}")
        st.dataframe(pd.DataFrame(matrix, index=["Actual: no disease", "Actual: disease"],
                                  columns=["Predicted: no disease", "Predicted: disease"]))


def show_roc_visualizations(actual: np.ndarray, probabilities: dict[str, np.ndarray], selected: str) -> None:
    curves = {name: roc_curve(actual, values)[:2] for name, values in probabilities.items()}
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for name, (fpr, tpr) in curves.items():
            area = roc_auc_score(actual, probabilities[name])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} · AUC {area:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line={"dash": "dash", "color": "gray"}))
        fig.update_layout(title="ROC curves · all ten algorithms", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=480)
        st.plotly_chart(fig, use_container_width=True)

        fpr, tpr = curves[selected]
        indices = np.unique(np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int))
        frames = [go.Frame(name=str(i), data=[go.Scatter(x=fpr[:i + 1], y=tpr[:i + 1])], traces=[0])
                  for i in indices]
        animated = go.Figure(data=[go.Scatter(x=fpr[:1], y=tpr[:1], mode="lines+markers",
                                              name=selected, line={"color": "#0F766E", "width": 3}),
                                  go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                             line={"dash": "dash", "color": "gray"})], frames=frames)
        animated.update_layout(title=f"Animated ROC curve · {selected}", xaxis_title="False positive rate",
                               yaxis_title="True positive rate", height=460,
                               updatemenus=[{"type": "buttons", "showactive": False,
                                             "buttons": [{"label": "▶ Play", "method": "animate",
                                                          "args": [None, {"frame": {"duration": 180, "redraw": True},
                                                                          "transition": {"duration": 80}}]}]}],
                               sliders=[{
                                   "currentvalue": {"prefix": "Threshold step: "},
                                   "steps": [
                                       {
                                           "label": frame.name,
                                           "method": "animate",
                                           "args": [
                                               [frame.name],
                                               {
                                                   "frame": {"duration": 0, "redraw": True},
                                                   "transition": {"duration": 0},
                                               },
                                           ],
                                       }
                                       for frame in frames
                                   ],
                               }])
        st.caption("Play to reveal the selected model's ROC curve as the decision threshold moves. A curve closer to the upper-left usually separates the classes better.")
        st.plotly_chart(animated, use_container_width=True, key="cardio-roc-animation")
    else:
        grid = np.linspace(0, 1, 101)
        table = {name: np.interp(grid, fpr, tpr) for name, (fpr, tpr) in curves.items()}
        st.line_chart(pd.DataFrame(table, index=grid).rename_axis("False positive rate"))
        st.info("Install Plotly to enable the animated ROC curve.")


def show_cardio_eda(frame: pd.DataFrame, key: str) -> None:
    try:
        features, target = prepare_cardio_frame(frame)
    except Exception as exc:
        st.warning(f"EDA is not available until the dataset fields are valid: {exc}")
        return
    st.subheader("Explore the cardiovascular dataset")
    left, right = st.columns(2)
    with left:
        counts = target.value_counts().sort_index().rename(index={0: "No disease", 1: "Disease"})
        st.write("**Target balance**")
        st.bar_chart(counts)
    with right:
        st.write("**Missing values in the uploaded file**")
        missing = frame.isna().sum().sort_values(ascending=False).head(12)
        st.bar_chart(missing[missing > 0] if (missing > 0).any() else pd.Series({"No missing values": 0}))
    numeric = features[CARDIO_NUMERIC].copy()
    numeric["cardio"] = target.to_numpy()
    if PLOTLY_AVAILABLE:
        corr = numeric.corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation of numeric features and outcome")
        st.plotly_chart(heat, use_container_width=True, key=f"{key}-corr")
        plots = st.columns(2)
        sample = features.copy()
        sample["cardio"] = target.map({0: "No disease", 1: "Disease"}).to_numpy()
        if len(sample) > 6000:
            sample = sample.sample(6000, random_state=42)
        with plots[0]:
            fig = px.histogram(sample, x="age_years", color="cardio", barmode="overlay", opacity=0.65,
                               title="Age distribution by outcome")
            st.plotly_chart(fig, use_container_width=True, key=f"{key}-age")
        with plots[1]:
            fig = px.scatter(sample, x="age_years", y="ap_hi", color="cardio", opacity=0.55,
                             title="Age and systolic blood pressure")
            st.plotly_chart(fig, use_container_width=True, key=f"{key}-scatter")
    else:
        st.line_chart(numeric.corr(numeric_only=True))
        st.line_chart(numeric[["age_years", "ap_hi", "ap_lo"]])


def classification_page():
    st.title("Classification · ten algorithms")
    st.write("Upload labeled cardiovascular records to compare all ten algorithms from both notebook sections. The file must contain a `cardio` target with values 0 and 1.")
    uploaded = st.file_uploader("Labeled CardioSense dataset (CSV or Excel)", type=["csv", "xlsx"], key="cardio-upload")
    if uploaded is None:
        st.markdown("**Part A:** Logistic Regression · K-Nearest Neighbors · Gaussian Naive Bayes · Decision Tree · SVM")
        st.markdown("**Part B:** Random Forest · AdaBoost · Gradient Boosting · Bagging · MLP Neural Network")
        st.caption("Expected fields: age (days or years), gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio. An `id` field is ignored.")
        return
    try:
        raw_bytes = uploaded.getvalue()
        signature = (uploaded.name, hash(raw_bytes))
        if st.session_state.get("cardio_signature") != signature:
            st.session_state.pop("cardio_result", None)
            st.session_state["cardio_signature"] = signature
        if uploaded.name.lower().endswith(".csv"):
            frame = pd.read_csv(io.BytesIO(raw_bytes), sep=";")
            if frame.shape[1] == 1:
                frame = pd.read_csv(io.BytesIO(raw_bytes))
        else:
            frame = pd.read_excel(io.BytesIO(raw_bytes))
        st.write(f"Loaded **{len(frame):,} records**. Evaluation uses a reproducible stratified sample of up to 5,000 valid records for responsive training.")
        with st.expander("Exploratory data analysis", expanded=True):
            show_cardio_eda(frame, "cardio-eda")
        if st.button("Evaluate 10 classifiers", type="primary"):
            with st.spinner("Cleaning the records and training all 10 classifiers…"):
                st.session_state["cardio_result"] = evaluate_cardio_models(frame)
        result = st.session_state.get("cardio_result")
        if result:
            scores = result["metrics"]
            st.success(f"Evaluated {result['valid_rows']:,} held-out records from {result['total_rows']:,} sampled, cleaned records.")
            st.subheader("Consolidated comparison · all ten algorithms")
            st.caption("Includes Accuracy, Precision, Recall, weighted F1, and ROC-AUC. This CardioSense task is binary, so ROC-AUC uses the disease-class probability.")
            st.dataframe(scores.style.format("{:.3f}"), use_container_width=True)
            metric = st.selectbox("Compare algorithms by", list(scores.columns), key="cardio-metric")
            if PLOTLY_AVAILABLE:
                chart = px.bar(scores.reset_index(), x="Algorithm", y=metric, color="Algorithm", text_auto=".3f",
                               title=f"Classifier comparison · {metric}", color_discrete_sequence=px.colors.qualitative.Safe)
                chart.update_layout(showlegend=False, xaxis_title=None, height=380)
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.bar_chart(scores[metric])
            selected = st.selectbox("Inspect classifier", CARDIO_MODELS, key="cardio-selected-model")
            threshold = st.slider("Decision threshold for disease class", 0.05, 0.95, 0.50, 0.01,
                                  help="A lower threshold predicts the positive class more often; adjust it to see the confusion matrix change.")
            probability = result["probabilities"][selected]
            prediction = (probability >= threshold).astype(int)
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(result['actual'], prediction):.3f}")
            col2.metric("Recall", f"{recall_score(result['actual'], prediction, zero_division=0):.3f}")
            col3.metric("F1 (weighted)", f"{f1_score(result['actual'], prediction, average='weighted'):.3f}")
            show_confusion_matrix(result["actual"], prediction, selected, "cardio-confusion-matrix")
            show_roc_visualizations(result["actual"], result["probabilities"], selected)
    except Exception as exc:
        st.error(f"Could not evaluate this classification file: {exc}")


def info_card(title: str, body: str):
    with st.container(border=True):
        st.subheader(title)
        st.write(body)


FINRISK_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"


def prepare_finrisk(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    required = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent",
                "loan_grade", "loan_amnt", "loan_int_rate", "loan_status", "loan_percent_income",
                "cb_person_default_on_file", "cb_person_cred_hist_length"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("FinRisk file is missing columns: " + ", ".join(missing))
    df = frame[required].copy().drop_duplicates()
    df["person_age"] = pd.to_numeric(df["person_age"], errors="coerce")
    df["person_emp_length"] = pd.to_numeric(df["person_emp_length"], errors="coerce")
    df["loan_int_rate"] = pd.to_numeric(df["loan_int_rate"], errors="coerce")
    df = df.loc[(df["person_age"] <= 100) & (df["person_emp_length"].isna() |
                  (df["person_emp_length"] <= df["person_age"] - 14))].copy()
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df.groupby("loan_grade")["loan_int_rate"].transform("median"))
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    df = df.dropna(subset=required).copy()
    if len(df) < 20:
        raise ValueError("At least 20 complete borrower records are required.")
    df["log_income"] = np.log1p(pd.to_numeric(df["person_income"], errors="coerce").clip(lower=0))
    df["log_loan_amnt"] = np.log1p(pd.to_numeric(df["loan_amnt"], errors="coerce").clip(lower=0))
    df["credit_hist_ratio"] = pd.to_numeric(df["cb_person_cred_hist_length"], errors="coerce") / df["person_age"]
    df["grade_num"] = df["loan_grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    df["default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    continuous = ["person_age", "log_income", "person_emp_length", "log_loan_amnt", "loan_int_rate",
                  "loan_percent_income", "cb_person_cred_hist_length", "credit_hist_ratio", "grade_num"]
    dummies = pd.get_dummies(df[["person_home_ownership", "loan_intent"]], dtype=int)
    scaled = pd.DataFrame(StandardScaler().fit_transform(df[continuous]), columns=continuous, index=df.index)
    features = pd.concat([scaled, df[["default_on_file"]], dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    valid = features.notna().all(axis=1)
    labels = pd.to_numeric(df.loc[valid, "loan_status"], errors="coerce").astype(int)
    return features.loc[valid].reset_index(drop=True), df.loc[valid].reset_index(drop=True), labels.reset_index(drop=True)


def analyze_finrisk(frame: pd.DataFrame) -> dict:
    features, clean, default_label = prepare_finrisk(frame)
    if len(features) > 10_000:
        idx = np.random.RandomState(42).choice(len(features), 10_000, replace=False)
        features, clean, default_label = features.iloc[idx].reset_index(drop=True), clean.iloc[idx].reset_index(drop=True), default_label.iloc[idx].reset_index(drop=True)
    matrix = features.to_numpy(dtype=float)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(matrix), min(3000, len(matrix)), replace=False)
    sweep, labels_by_k = [], {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(matrix)
        labels = model.labels_
        labels_by_k[k] = labels
        sweep.append({"k": k, "Inertia": model.inertia_,
                      "Silhouette Score": silhouette_score(matrix[sample_idx], labels[sample_idx]),
                      "Davies-Bouldin Index": davies_bouldin_score(matrix, labels),
                      "Calinski-Harabasz Index": calinski_harabasz_score(matrix, labels)})
    scores = pd.DataFrame(sweep)
    best_k = int(scores.loc[scores["Silhouette Score"].idxmax(), "k"])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)
    continuous_profile = clean[["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
                                "loan_percent_income", "cb_person_cred_hist_length"]].copy()
    return {"features": features, "clean": clean, "default": default_label, "scores": scores,
            "labels_by_k": labels_by_k, "best_k": best_k, "coords": coords,
            "profile_source": continuous_profile, "variance": pca.explained_variance_ratio_}


def show_finrisk_eda(frame: pd.DataFrame) -> None:
    st.subheader("FinRisk exploratory data analysis")
    st.write(f"Raw shape: **{frame.shape[0]:,} rows × {frame.shape[1]} columns** · duplicates: **{frame.duplicated().sum():,}**")
    left, right = st.columns(2)
    with left:
        st.write("**Missing values**")
        missing = frame.isna().sum().sort_values(ascending=False).head(12)
        st.bar_chart(missing[missing > 0] if (missing > 0).any() else pd.Series({"No missing values": 0}))
    with right:
        if "loan_status" in frame:
            st.write("**Observed loan status** (held out of clustering features)")
            st.bar_chart(frame["loan_status"].value_counts().sort_index())
    if "person_income" in frame:
        income = pd.to_numeric(frame["person_income"], errors="coerce").dropna()
        eda = pd.DataFrame({"Income": income.clip(upper=income.quantile(.99)), "Log income": np.log1p(income.clip(lower=0))})
        if PLOTLY_AVAILABLE:
            hist = px.histogram(eda, x="Income", nbins=40, title="Income distribution (capped at 99th percentile for display)")
            st.plotly_chart(hist, use_container_width=True, key="finrisk-income-hist")
            numeric_cols = [c for c in ["person_age", "person_income", "person_emp_length", "loan_amnt",
                                        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"] if c in frame]
            if len(numeric_cols) > 1:
                corr = px.imshow(frame[numeric_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Borrower feature correlations")
                st.plotly_chart(corr, use_container_width=True, key="finrisk-eda-corr")


def show_kmeans_results(result: dict, key: str) -> None:
    scores = result["scores"]
    st.caption("Compare k using Silhouette (higher is better), Davies–Bouldin (lower is better), and Calinski–Harabasz (higher is better). Inertia is shown for the elbow method.")
    for row in range(2):
        cols = st.columns(2)
        metrics = [("Inertia", "Elbow curve · inertia by k"),
                   ("Silhouette Score", "Silhouette score by k"),
                   ("Davies-Bouldin Index", "Davies–Bouldin index by k"),
                   ("Calinski-Harabasz Index", "Calinski–Harabasz index by k")]
        for col, (metric, title) in zip(cols, metrics[row * 2:(row + 1) * 2]):
            with col:
                if PLOTLY_AVAILABLE:
                    fig = px.line(scores, x="k", y=metric, markers=True, title=title)
                    st.plotly_chart(fig, use_container_width=True, key=f"{key}-{row}-{metric}")
                else:
                    st.line_chart(scores.set_index("k")[metric])
    st.dataframe(scores.style.format({c: "{:.3f}" for c in scores.columns if c != "k"}), use_container_width=True)
    selected_k = st.selectbox("Inspect K-Means result for k", list(range(2, 9)), index=result["best_k"] - 2,
                              key=f"{key}-k-selected")
    labels = result["labels_by_k"][selected_k]
    sizes = pd.Series(labels).value_counts().sort_index().rename_axis("Cluster").rename("Borrowers")
    st.write(f"Silhouette-selected k: **{result['best_k']}**. Current display: **k={selected_k}**.")
    st.bar_chart(sizes)
    profile = result["profile_source"].assign(Cluster=labels).groupby("Cluster").mean(numeric_only=True)
    st.write("**Cluster profiles · mean in original units**")
    st.dataframe(profile.style.format("{:,.2f}"), use_container_width=True)
    default_rates = result["default"].groupby(labels).mean().rename("Observed default rate").to_frame()
    st.write("**Post-hoc default rates** (loan_status was not used to form clusters)")
    st.dataframe(default_rates.style.format("{:.1%}"), use_container_width=True)
    sample = np.linspace(0, len(labels) - 1, min(2500, len(labels))).astype(int)
    plot_data = pd.DataFrame({"PC1": result["coords"][sample, 0], "PC2": result["coords"][sample, 1],
                              "Cluster": labels[sample].astype(str), "k": "k=" + str(selected_k)})
    if PLOTLY_AVAILABLE:
        fig = px.scatter(plot_data, x="PC1", y="PC2", color="Cluster", animation_frame="k",
                         title=f"K-Means clusters in PCA space · PC variance {result['variance'].sum():.1%}",
                         opacity=.65, color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key=f"{key}-pca")
        # Animate through each k considered by the notebook, not just the selected value.
        frames_data = []
        for kval, labs in result["labels_by_k"].items():
            part = pd.DataFrame({"PC1": result["coords"][sample, 0], "PC2": result["coords"][sample, 1],
                                 "Cluster": labs[sample].astype(str), "k": f"k={kval}"})
            frames_data.append(part)
        combined = pd.concat(frames_data, ignore_index=True)
        animated = px.scatter(combined, x="PC1", y="PC2", color="Cluster", animation_frame="k",
                              title="Animated cluster assignments as k changes", opacity=.65,
                              range_x=[float(result["coords"][:, 0].min()), float(result["coords"][:, 0].max())],
                              range_y=[float(result["coords"][:, 1].min()), float(result["coords"][:, 1].max())])
        animated.update_layout(height=520)
        st.caption("Play to see how the same PCA projection is divided as the number of clusters changes.")
        st.plotly_chart(animated, use_container_width=True, key=f"{key}-k-animation")
    else:
        st.scatter_chart(plot_data, x="PC1", y="PC2", color="Cluster")


def show_agglomerative_iris(key: str, uploaded_frame: pd.DataFrame | None = None) -> None:
    if uploaded_frame is None:
        iris = load_iris(as_frame=True)
        frame = iris.frame.rename(columns={"sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width",
                                           "petal length (cm)": "petal_length", "petal width (cm)": "petal_width",
                                           "target": "species_code"})
        frame["species"] = frame["species_code"].map(dict(enumerate(iris.target_names)))
    else:
        frame = uploaded_frame.copy()
        normalized = {re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_"): col
                      for col in frame.columns}
        aliases = {
            "sepal_length": ["sepal_length", "sepal_length_cm", "sepallengthcm"],
            "sepal_width": ["sepal_width", "sepal_width_cm", "sepalwidthcm"],
            "petal_length": ["petal_length", "petal_length_cm", "petallengthcm"],
            "petal_width": ["petal_width", "petal_width_cm", "petalwidthcm"],
        }
        rename = {}
        for target, names in aliases.items():
            if target not in frame.columns:
                for alias in names:
                    if alias in normalized:
                        rename[normalized[alias]] = target
                        break
        frame = frame.rename(columns=rename)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise ValueError("Upload an Iris-style CSV with these measurement columns: " + ", ".join(features)
                         + f". Missing: {', '.join(missing)}. Columns found: {', '.join(map(str, frame.columns))}")
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=features).drop_duplicates().reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 complete rows are needed for Agglomerative clustering.")
    if len(frame) > 1000:
        frame = frame.sample(n=1000, random_state=42).reset_index(drop=True)
        st.info("Agglomerative clustering uses a reproducible 1,000-row sample for large uploads because its memory use grows rapidly with row count.")
    x = frame[features]
    scaled = StandardScaler().fit_transform(x)
    st.subheader("Iris exploratory data analysis")
    if "species" in frame.columns:
        st.write(f"**{len(frame)} rows**, **4 measurements**. Species labels are shown for EDA only and excluded from clustering.")
    else:
        st.write(f"**{len(frame)} rows**, **4 measurements**. Any extra columns, including labels, are excluded from clustering.")
    a, b = st.columns(2)
    with a:
        if "species" in frame.columns:
            st.write("**Species counts**")
            st.bar_chart(frame["species"].value_counts())
        else:
            st.write("**Species counts**")
            st.caption("No species column supplied; clustering uses measurements only.")
    with b:
        st.write("**Feature summary**")
        st.dataframe(x.describe().T.round(2), use_container_width=True)
    if PLOTLY_AVAILABLE:
        corr = px.imshow(x.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Iris feature correlations")
        st.plotly_chart(corr, use_container_width=True, key=f"{key}-corr")
        if "species" in frame.columns:
            pair = px.scatter(frame, x="petal_length", y="petal_width", color="species",
                              title="Petal measurements by known species (EDA only)")
            st.plotly_chart(pair, use_container_width=True, key=f"{key}-species-eda")
        else:
            pair = px.scatter(frame, x="petal_length", y="petal_width",
                              title="Petal measurements in uploaded rows")
            st.plotly_chart(pair, use_container_width=True, key=f"{key}-petal-eda")
    methods = ["ward", "complete", "average", "single"]
    labels_by_method, rows = {}, []
    for method in methods:
        labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(scaled)
        labels_by_method[method] = labels
        rows.append({"Linkage": method,
                     "Silhouette Score": silhouette_score(scaled, labels),
                     "Davies-Bouldin Index": davies_bouldin_score(scaled, labels),
                     "Calinski-Harabasz Index": calinski_harabasz_score(scaled, labels)})
    scores = pd.DataFrame(rows).sort_values("Silhouette Score", ascending=False)
    best = str(scores.iloc[0]["Linkage"])
    st.subheader("Agglomerative metrics by linkage · 3 clusters")
    st.caption("Silhouette and Calinski–Harabasz are better when higher; Davies–Bouldin is better when lower.")
    st.dataframe(scores.style.format({c: "{:.3f}" for c in scores.columns if c != "Linkage"}), use_container_width=True)
    if PLOTLY_AVAILABLE:
        for metric in ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]:
            score_chart = px.bar(scores, x="Linkage", y=metric, color="Linkage", title=f"{metric} by linkage")
            st.plotly_chart(score_chart, use_container_width=True, key=f"{key}-linkage-{metric}")
    selected = st.selectbox("Linkage to inspect", methods, index=methods.index(best), key=f"{key}-linkage")
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plot_data = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                              "Cluster": labels_by_method[selected].astype(str), "Linkage": selected})
    st.write(f"Highest silhouette in this run: **{best}**")
    st.bar_chart(pd.Series(labels_by_method[selected]).value_counts().sort_index().rename("Flowers"))
    st.subheader("Compare K-Means and Agglomerative on the same dataset")
    st.caption("Both algorithms below use the same scaled Iris measurements and three clusters. FinRisk scores above are dataset-specific and should not be compared directly with Iris scores.")
    kmeans_labels = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42).fit_predict(scaled)
    agglomerative_labels = labels_by_method[selected]
    comparison = pd.DataFrame([
        {"Algorithm": "K-Means", "Configuration": "k=3",
         "Silhouette Score": silhouette_score(scaled, kmeans_labels),
         "Davies-Bouldin Index": davies_bouldin_score(scaled, kmeans_labels),
         "Calinski-Harabasz Index": calinski_harabasz_score(scaled, kmeans_labels)},
        {"Algorithm": "Agglomerative", "Configuration": f"linkage={selected}, clusters=3",
         "Silhouette Score": silhouette_score(scaled, agglomerative_labels),
         "Davies-Bouldin Index": davies_bouldin_score(scaled, agglomerative_labels),
         "Calinski-Harabasz Index": calinski_harabasz_score(scaled, agglomerative_labels)},
    ])
    st.dataframe(comparison.style.format({c: "{:.3f}" for c in comparison.columns if c.endswith("Index") or c == "Silhouette Score"}),
                 use_container_width=True)
    metric_cols = st.columns(3)
    for col, metric in zip(metric_cols, ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]):
        with col:
            if PLOTLY_AVAILABLE:
                compare_chart = px.bar(comparison, x="Algorithm", y=metric, color="Algorithm",
                                       title=metric, text_auto=".3f")
                compare_chart.update_layout(showlegend=False)
                st.plotly_chart(compare_chart, use_container_width=True, key=f"{key}-algorithm-compare-{metric}")
            else:
                st.bar_chart(comparison.set_index("Algorithm")[[metric]])
    if PLOTLY_AVAILABLE:
        fig = px.scatter(plot_data, x="PC1", y="PC2", color="Cluster", title=f"Agglomerative clusters · {selected} linkage")
        st.plotly_chart(fig, use_container_width=True, key=f"{key}-pca")
        frames = []
        for method, labels in labels_by_method.items():
            part = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                                 "Cluster": labels.astype(str), "Linkage": method})
            frames.append(part)
        animated = px.scatter(pd.concat(frames), x="PC1", y="PC2", color="Cluster", animation_frame="Linkage",
                              title="Animated cluster assignments by linkage", range_x=[-4, 4], range_y=[-2, 2])
        st.caption("Play to compare how the four linkage rules group the same Iris flowers.")
        st.plotly_chart(animated, use_container_width=True, key=f"{key}-linkage-animation")
        z = linkage(scaled, method="ward")
        den = dendrogram(z, no_plot=True)
        dend = go.Figure(go.Scatter(x=den["icoord"][0], y=den["dcoord"][0], mode="lines", line={"color": "#0F766E"}))
        for xs, ys in zip(den["icoord"][1:], den["dcoord"][1:]):
            dend.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#0F766E"}, showlegend=False))
        dend.update_layout(title="Ward linkage dendrogram", xaxis_title="Flower leaves", yaxis_title="Merge distance", height=440)
        st.plotly_chart(dend, use_container_width=True, key=f"{key}-dendrogram")
    else:
        st.scatter_chart(plot_data, x="PC1", y="PC2", color="Cluster")


def clustering_page():
    st.title("Clustering")
    st.write("Both clustering notebooks are available here. Choose a workflow to explore its dataset, preprocessing, clustering results, and visualizations.")
    tabs = st.tabs(["K-Means · FinRisk notebook", "Agglomerative · Iris notebook"])
    with tabs[0]:
        st.subheader("K-Means · FinRisk borrower segmentation")
        st.caption("Source notebook: `clustering_kmeans.ipynb` · Uses credit-risk features; loan status is reserved for post-clustering analysis.")
        st.markdown("**Goal:** segment borrowers. `loan_status` is held aside and only checked after clustering.")
        uploaded = st.file_uploader("FinRisk dataset CSV (optional if project data/ is available)", type=["csv"], key="finrisk-upload")
        try:
            if uploaded is not None:
                raw_bytes = uploaded.getvalue()
                frame = pd.read_csv(io.BytesIO(raw_bytes))
                signature = (uploaded.name, hash(raw_bytes))
            elif FINRISK_FILE.exists():
                frame = pd.read_csv(FINRISK_FILE)
                signature = (str(FINRISK_FILE), FINRISK_FILE.stat().st_mtime_ns)
                st.caption(f"Using project dataset: `{FINRISK_FILE.name}`")
            else:
                st.info("Upload `credit_risk_dataset.csv` to run FinRisk clustering.")
                frame = None
                signature = None
            if frame is not None:
                with st.expander("EDA · dataset audit and distributions", expanded=True):
                    show_finrisk_eda(frame)
                if st.session_state.get("finrisk_signature") != signature:
                    st.session_state.pop("finrisk_result", None)
                    st.session_state["finrisk_signature"] = signature
                if st.button("Run K-Means and choose k", type="primary", key="finrisk-run"):
                    with st.spinner("Preparing FinRisk features and evaluating k=2 through 8…"):
                        st.session_state["finrisk_result"] = analyze_finrisk(frame)
                result = st.session_state.get("finrisk_result")
                if result:
                    show_kmeans_results(result, "finrisk-results")
        except Exception as exc:
            st.error(f"Could not run FinRisk clustering: {exc}")
    with tabs[1]:
        st.subheader("Agglomerative Hierarchical Clustering · Iris")
        st.caption("Source notebook: `agglomerative_hierarchical_clustering.ipynb` · Compares Ward, complete, average, and single linkage.")
        st.markdown("**Goal:** group Iris flowers using four scaled measurements; the known species label is used for EDA only.")
        st.markdown("Upload an optional CSV with `sepal_length`, `sepal_width`, `petal_length`, and `petal_width`. Without an upload, the app uses scikit-learn's built-in Iris dataset.")
        iris_upload = st.file_uploader("Iris-style dataset CSV (optional)", type=["csv"], key="iris-cluster-upload")
        try:
            uploaded_iris = pd.read_csv(io.BytesIO(iris_upload.getvalue())) if iris_upload is not None else None
            show_agglomerative_iris("iris-agglomerative", uploaded_iris)
        except Exception as exc:
            st.error(f"Could not run Agglomerative clustering: {exc}")


def navigate_to(track: str) -> None:
    st.session_state["main_nav"] = track


if "main_nav" not in st.session_state:
    st.session_state["main_nav"] = "Overview"

with st.sidebar:
    st.markdown("# 🧪 ML Studio")
    st.caption("Machine Learning Capstone")
    page = st.radio("Choose a track", ["Overview", "Regression", "Classification", "Clustering"],
                    key="main_nav", label_visibility="collapsed")
    st.divider()
    st.caption("Upload datasets to explore regression, classification, and clustering workflows.")

if page == "Overview":
    st.title("Machine Learning Capstone")
    st.markdown("### Learn by exploring three machine learning tasks")
    st.write("Choose a track to open its working page.")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.subheader("📈 Regression")
            st.write("Compare ten fare prediction algorithms, inspect errors, and explore the flight data.")
            st.button("Open Regression", on_click=navigate_to, args=("Regression",),
                      key="overview-regression", use_container_width=True)
    with c2:
        with st.container(border=True):
            st.subheader("🩺 Classification")
            st.write("Explore cardiovascular data and compare ten classifiers with ROC curves and confusion matrices.")
            st.button("Open Classification", on_click=navigate_to, args=("Classification",),
                      key="overview-classification", use_container_width=True)
    with c3:
        with st.container(border=True):
            st.subheader("🧩 Clustering")
            st.write("Run K-Means borrower segmentation and compare Agglomerative clustering strategies on Iris.")
            st.button("Open Clustering", on_click=navigate_to, args=("Clustering",),
                      key="overview-clustering", use_container_width=True)
elif page == "Regression":
    regression_page()
elif page == "Classification":
    classification_page()
else:
    clustering_page()
