import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# -------------------- FIRST STREAMLIT CALL --------------------
st.set_page_config(page_title="ML Studio: Predict & Forecast", page_icon="🤖", layout="wide")

# -------------------- Silence noisy warnings --------------------
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", message=".*'M' is deprecated.*", category=FutureWarning)

# -------------------- Light-only theme --------------------
pio.templates.default = "plotly_white"

def get_css() -> str:
    return """
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 4.5rem !important; padding-bottom: 3rem; }
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 600px at 10% 10%, #ffffff 0%, #f5f7fb 50%, #eef2f7 100%) !important; color: #1f2335;
}
h1, h2, h3, h4, h5, h6 { letter-spacing: 0.2px; color:#1f2335; }
.glass { background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85)); border: 1px solid rgba(0,0,0,0.08); border-radius: 18px; padding: 18px 18px; box-shadow: 0 10px 30px rgba(31,35,53,0.10), inset 0 1px 0 rgba(255,255,255,0.6); }
.dataframe thead tr th { background: #eef2f7 !important; color: #1f2335 !important; border-bottom: 1px solid rgba(0,0,0,0.06) !important; }
.dataframe tbody tr:hover { background: rgba(31,35,53,0.04) !important; }
.stButton>button { border-radius: 12px; border: 1px solid rgba(0,0,0,0.12); background: linear-gradient(180deg, rgba(31,35,53,0.06), rgba(31,35,53,0.03)); transition: all .2s ease; color: #1f2335; }
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(31,35,53,0.15); }
.footer { opacity: 0.8; font-size: 12px; text-align: center; margin-top: 32px; color: #4b4f6a; }
.badge  { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; border:1px solid rgba(0,0,0,0.12); background:rgba(0,0,0,0.04); color:#1f2335; }
"""

st.markdown(f"<style>{get_css()}</style>", unsafe_allow_html=True)

# Header
logo_path = os.path.join("assets", "logo.png")
cols = st.columns([1,6,2])
with cols[0]:
    if os.path.exists(logo_path):
        st.image(logo_path, width=84)
with cols[1]:
    st.markdown("<h1 style='margin-bottom:0'>Hemraj ML Studio</h1>", unsafe_allow_html=True)
    st.markdown("<div class='badge'>Predict • Explain • Forecast</div>", unsafe_allow_html=True)
with cols[2]:
    pass
st.markdown("<div class='glass'>", unsafe_allow_html=True)

# -------------------- Helpers --------------------
PREFERRED_TARGETS = [
    "Chite (KG/Day)","Mota Kuro (KG/Day)","Husk Production (Per Day/Kg) #1",
    "Rice Load (KG/Day)","Steam Production (KG/Day)","Rubber Rolls Used (PC)",
    "Broken (For Mota Kuro - KG/Day)","Bran Production (Per Bag/Kg) #1",
    "Store Issue Expense","Paddy Chalai (KG/Day)",
]

@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    if isinstance(file_or_path, str):
        return pd.read_csv(file_or_path)
    return pd.read_csv(file_or_path)

def num_cols(df): 
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def coerce_object_numerics(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() >= 0.6:
                df[c] = coerced
    return df

def find_datetime_cols(df: pd.DataFrame):
    """Quiet, robust detector (no spammy warnings)."""
    out = []
    for c in df.columns:
        name_is_datey = "date" in str(c).lower()
        if not name_is_datey and not pd.api.types.is_object_dtype(df[c]):
            continue
        s = df[c].astype(str).str.strip()
        parsed = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            p = pd.to_datetime(s, errors="coerce", format=fmt)
            if p.notna().mean() >= 0.7:
                parsed = p
                break
        if parsed is None:
            p = pd.to_datetime(s, errors="coerce", dayfirst=True)
            if p.notna().mean() >= 0.7:
                parsed = p
        if parsed is not None:
            out.append(c)
    return out

def make_ohe():
    """OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2

def ml_pipeline(X: pd.DataFrame):
    """Preprocess + RandomForest regressor."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categor = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                             ("scale", StandardScaler())]), numeric))
    if categor:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                             ("oh", make_ohe())]), categor))
    pre = ColumnTransformer(transformers=transformers, remainder="drop") if transformers else "drop"
    model = RandomForestRegressor(n_estimators=250, random_state=42)
    if transformers:
        return Pipeline([("prep", pre), ("model", model)])
    else:
        # If no preprocessing needed (rare), still return a pipeline
        return Pipeline([("model", model)])

def download_df_button(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=file_name, mime="text/csv")

def _prepare_series_for_forecast(series, desired_freq=None):
    """
    Regularize a time series: set explicit freq, fill gaps, remove NaNs.
    - If desired_freq given ("D","W","ME"), resample to that freq and asfreq().
    - Else infer a reasonable freq; fallback to daily.
    """
    s = series.copy()
    if desired_freq:
        s = s.resample(desired_freq).mean()
        s = s.asfreq(desired_freq)
    else:
        inferred = s.index.inferred_freq
        if not inferred:
            try:
                inferred = pd.infer_freq(s.index)
            except Exception:
                inferred = None
        s = s.asfreq(inferred or "D")
    s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
    return s.dropna()

# -------------------- Data load --------------------
st.sidebar.title(" Data")
demo_path = os.path.join("demo_data", "data_ana.csv")
use_demo = os.path.exists(demo_path) and st.sidebar.toggle("Use bundled demo CSV", value=True, key="use_demo")
uploaded = None
if not use_demo:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="upload_csv")

if uploaded is not None and not use_demo:
    raw = load_csv(uploaded)
elif use_demo and os.path.exists(demo_path):
    raw = load_csv(demo_path)
else:
    st.info("Upload a CSV (or enable 'Use bundled demo CSV').")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- Custom cleanup: normalize Date and drop unwanted columns ---
drop_cols = [
    "Bran Production ( Per Bag /Kg) #2",
    "Bran Production ( Per Bag /Kg) #3",
    "Bran Production ( Per Bag /Kg) #4",
    "Bran Production ( Per Bag /Kg) #5",
    "Bran Production ( Per Bag /Kg) #6",
    "Broken Production ( Per Bag /Kg) #2",
    "Broken Production ( Per Bag /Kg) #3",
    "Broken Production ( Per Bag /Kg) #4",
    "Broken Production ( Per Bag /Kg) #5",
    "Broken Production ( Per Bag Kg) #6",
    "Broken Production ( Per Bag /Kg) #7",
    "Broken Production ( Per Bag /Kg) #8",
    "Broken Production ( Per Bag /Kg) #9",
    "Broken Production ( Per Bag /Kg) #10",
    "Broken Production ( Per Bag /Kg) #11",
    "Broken Production ( Per Bag /Kg) #12",
    "Husk Production (Per day/Kg) #2",
    "Husk Production (Perday/Kg) #3",
    "Husk Production (Per Day/Kg) #4",
]
existing_drops = [c for c in drop_cols if c in raw.columns]
if existing_drops:
    raw = raw.drop(columns=existing_drops)

if "Date" in raw.columns:
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.sort_values(by="Date")
# --- end custom cleanup ---

df = coerce_object_numerics(raw)
st.success(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
with st.expander(" Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    " Focused Targets (Multi-Input)",
    " One-Input → Others",
    " Forecast (with resampling)",
    " Quick EDA",
])

# -------------------- Tab 1: Multi-Input (RandomForest only) --------------------
with tab1:
    st.subheader(" Train RandomForest models for your key targets (multiple input features)")
    existing_pref = [c for c in PREFERRED_TARGETS if c in df.columns]
    defaults = existing_pref if existing_pref else ([num_cols(df)[0]] if num_cols(df) else [])
    targets = st.multiselect("Select target columns", options=df.columns.tolist(), default=defaults, key="targets_multi")

    candidates = [c for c in df.columns if c not in targets]
    X_cols = st.multiselect("Select input features (X)", options=candidates,
                            default=candidates[: min(15, len(candidates))], key="xcols_multi")

    st.caption("Model: RandomForestRegressor (n_estimators=250)")

    if st.button("Train all target models", key="train_multi"):
        if not targets or not X_cols:
            st.error("Please select at least one target and one input feature.")
        else:
            trained = {}
            for y in targets:
                try:
                    XY = df[X_cols + [y]].copy().dropna(subset=[y])
                    if len(XY) >= 5 and X_cols:
                        X_all, y_all = XY[X_cols], XY[y]
                        pipe = ml_pipeline(X_all)  # RandomForest inside
                        pipe.fit(X_all, y_all)
                        trained[y] = pipe
                except Exception:
                    pass
            if trained:
                st.success(f"Trained {len(trained)} target model(s).")
            else:
                st.warning("No models trained. Check your selections.")
            st.session_state["multi_models"] = {"pipes": trained, "X_cols": X_cols, "targets": targets}

    mm = st.session_state.get("multi_models")
    if mm and mm.get("pipes"):
        st.divider()
        st.markdown("####  Predict new rows")
        row = {}
        for c in mm["X_cols"]:
            if pd.api.types.is_numeric_dtype(df[c]):
                row[c] = st.number_input(f"{c}", value=float(np.nan_to_num(df[c].median() if df[c].notna().any() else 0.0)), key=f"num_{c}_multi")
            else:
                opts = sorted(df[c].dropna().unique().tolist()) if df[c].dropna().nunique()>0 else ["N/A"]
                row[c] = st.selectbox(f"{c}", options=opts, index=0, key=f"cat_{c}_multi")
        if st.button("Predict targets", key="predict_multi"):
            row_df, preds = pd.DataFrame([row]), {}
            for tgt, pipe in mm["pipes"].items():
                try: preds[tgt] = float(pipe.predict(row_df)[0])
                except Exception: preds[tgt] = np.nan
            out = pd.DataFrame([preds])
            st.dataframe(out, use_container_width=True)
            download_df_button(out, " Download predictions CSV", "predictions.csv")

# -------------------- Tab 2: One-Input → Others (RandomForest only) --------------------
with tab2:
    st.subheader(" Pick ONE known column and predict all other numeric columns (RandomForest)")
    nnum = num_cols(df)
    if len(nnum) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        input_col = st.selectbox("Known input column", nnum, index=0, key="known_input_col")
        st.caption("Model: RandomForestRegressor (n_estimators=250)")

        if st.button("Train per-target models", key="train_one_input"):
            X_all = df[[input_col]].values
            targets = [c for c in nnum if c != input_col]
            models = {}
            for tgt in targets:
                y_all = df[tgt].values
                mask = (~pd.isna(X_all).ravel()) & (~pd.isna(y_all))
                X, y = X_all[mask].reshape(-1, 1), y_all[mask]
                if len(y) < 5: 
                    continue
                mdl = RandomForestRegressor(n_estimators=250, random_state=42)
                mdl.fit(X, y)  # fit on all data (no holdout)
                models[tgt] = mdl
            if models:
                st.session_state["one_input_models"] = models
                st.session_state["one_input_col"] = input_col
                st.success(f"Trained {len(models)} models using input: {input_col}")
            else:
                st.warning("Could not train any models. Check for NaNs or insufficient data.")

        if st.session_state.get("one_input_models") is not None and st.session_state.get("one_input_col") == input_col:
            default_val = float(np.nan_to_num(np.nanmean(df[input_col].values), nan=0.0))
            val = st.number_input(f"Enter value for {input_col}", value=default_val, key="val_one_input")
            if st.button("Predict other columns", key="predict_one_input"):
                preds = {}
                for tgt, mdl in st.session_state["one_input_models"].items():
                    try: preds[tgt] = float(mdl.predict(np.array([[val]]))[0])
                    except Exception: preds[tgt] = np.nan
                out = pd.DataFrame([preds])
                st.dataframe(out, use_container_width=True)
                download_df_button(out, " Download predictions CSV", "one_input_predictions.csv")
        else:
            st.info("Select an input column and click **Train per-target models** to enable predictions.")

# -------------------- Tab 3: Forecast (with resampling) --------------------
with tab3:
    st.subheader(" Forecast with optional resampling")
    dt_candidates = find_datetime_cols(df)
    if len(dt_candidates) == 0:
        st.warning("No datetime-like columns detected. We'll use the row index as time.")
        time_col, df_ts, use_resample = None, df.copy(), False
        df_ts["__time__"] = np.arange(len(df_ts))
    else:
        time_col = st.selectbox("Datetime column", dt_candidates, index=0, key="dt_col")
        df_ts = df.copy()
        df_ts[time_col] = pd.to_datetime(df_ts[time_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[time_col]).sort_values(time_col)
        use_resample = st.checkbox("Resample time series", value=True, key="resample_chk")

    t_candidates = num_cols(df_ts)
    if "__time__" in t_candidates: t_candidates.remove("__time__")
    if len(t_candidates) == 0:
        st.info("No numeric columns available to forecast.")
    else:
        target = st.selectbox("Target to forecast", t_candidates, index=0, key="target_forecast")
        freq_map = {"Daily":"D", "Weekly":"W", "Monthly":"ME"}  # Month-End
        freq_choice = st.selectbox("Frequency", list(freq_map.keys()), index=2, key="freq_choice")
        horizon = st.slider("Horizon (steps)", 1, 60, 12, key="horizon")
        show_baseline = st.checkbox("Show rolling mean baseline", value=True, key="show_baseline")

        if st.button("Build forecast", key="build_fc"):
            if time_col is None:
                s = df_ts[target].dropna().reset_index(drop=True)
                if len(s) < 5:
                    st.error("Not enough data points.")
                else:
                    X = np.arange(len(s)).reshape(-1,1); y = s.values
                    lr = LinearRegression().fit(X, y)
                    y_hat = lr.predict(X)
                    last_idx = len(s) - 1
                    f_idx = np.arange(last_idx+1, last_idx+1+horizon).reshape(-1,1)
                    f_pred = lr.predict(f_idx)
                    plot_df = pd.DataFrame({"x": np.r_[np.arange(len(s)), np.arange(last_idx+1, last_idx+1+horizon)],
                                            "y": np.r_[y, f_pred],
                                            "type": ["History"]*len(y)+["Forecast"]*len(f_pred)})
                    st.plotly_chart(px.line(plot_df, x="x", y="y", color="type", markers=True,
                                            title=f"Forecast for {target}"), use_container_width=True)
                    ftable = pd.DataFrame({"step": np.arange(1,horizon+1), "forecast": f_pred})
                    st.dataframe(ftable, use_container_width=True); download_df_button(ftable, " Download forecast CSV", "forecast.csv")
            else:
                df_ts = df_ts[[time_col, target]].dropna().set_index(time_col).sort_index()
                if use_resample:
                    series = _prepare_series_for_forecast(df_ts[target], desired_freq=freq_map[freq_choice])
                    freq_used = freq_map[freq_choice]
                else:
                    series = _prepare_series_for_forecast(df_ts[target], desired_freq=None)
                    freq_used = series.index.freqstr or "D"

                if len(series) < 5:
                    st.error("Not enough data points after resampling/filtering.")
                else:
                    s = series
                    idx = np.arange(len(s))
                    X = {"t": idx}
                    if str(freq_used).startswith("M"):
                        X["month"] = [ts.month for ts in s.index]
                    elif str(freq_used).startswith("D") or str(freq_used).startswith("B"):
                        X["dow"] = [ts.weekday() for ts in s.index]
                    elif str(freq_used).startswith("W"):
                        X["week"] = [ts.isocalendar().week for ts in s.index]
                    X = pd.DataFrame(X, index=s.index)

                    pre = ColumnTransformer([
                        ("num", StandardScaler(), ["t"]),
                        ("cat", make_ohe(), [c for c in X.columns if c!="t"])
                    ], remainder="drop")
                    pipe = Pipeline([("prep", pre), ("model", LinearRegression())])
                    pipe.fit(X, s.values); y_hat = pipe.predict(X)

                    future_index = pd.date_range(start=s.index[-1], periods=horizon+1, freq=freq_used)[1:]
                    Xf = {"t": np.arange(len(s), len(s)+horizon)}
                    if "month" in X.columns: Xf["month"] = [ts.month for ts in future_index]
                    if "dow"   in X.columns: Xf["dow"]   = [ts.weekday() for ts in future_index]
                    if "week"  in X.columns: Xf["week"]  = [ts.isocalendar().week for ts in future_index]
                    Xf = pd.DataFrame(Xf, index=future_index)
                    f_pred = pipe.predict(Xf)

                    hist_df = pd.DataFrame({"x": s.index, "y": s.values, "type": "History"})
                    fut_df  = pd.DataFrame({"x": future_index, "y": f_pred, "type": "Forecast"})
                    plot_df = pd.concat([hist_df, fut_df], ignore_index=True)
                    st.plotly_chart(px.line(plot_df, x="x", y="y", color="type", markers=True,
                                            title=f"Forecast for {target}"), use_container_width=True)
                    ftable = pd.DataFrame({"date": future_index, "forecast": f_pred})
                    st.dataframe(ftable, use_container_width=True); download_df_button(ftable, " Download forecast CSV", "forecast.csv")
                    if show_baseline:
                        roll = pd.Series(s.values).rolling(window=5, min_periods=1).mean().values
                        comp = pd.DataFrame({"x": s.index, "Actual": s.values, "Fitted": y_hat, "Rolling mean (w=5)": roll})
                        st.plotly_chart(px.line(comp, x="x", y=["Actual","Fitted","Rolling mean (w=5)"], markers=True,
                                                title="Model vs Baseline"), use_container_width=True)

# -------------------- Tab 4: EDA --------------------
with tab4:
    st.subheader(" Quick EDA")
    nnum = num_cols(df)
    if nnum:
        st.markdown("**Summary (numeric)**")
        st.dataframe(df[nnum].describe().T, use_container_width=True)
        st.markdown("**Correlation heatmap**")
        corr = df[nnum].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto=False, title="Correlation Heatmap"), use_container_width=True)
    else:
        st.info("No numeric columns detected.")

# Footer / close glass
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Built with Streamlit • scikit-learn • Plotly</div>", unsafe_allow_html=True)