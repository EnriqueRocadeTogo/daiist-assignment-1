"""
Assignment 1 — Gradio dashboard: fair value of central-Madrid flats.

Loads what train.ipynb saved in artifacts/ (the three trained models, the fitted
FeatureBuilder, the cleaned listings, metrics and training curves) and only runs
inference. Nothing in this file trains or refits anything.

    uv run python main.py app
"""

import json

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

import pipeline as P

ART = P.ARTIFACTS_DIR

# Validated categorical palette (first three slots are CVD-safe all-pairs) + chrome
C_SK, C_MAN, C_MOD = "#2a78d6", "#eb6834", "#1baf7a"
C_BASE, C_MUTED, C_GRID, C_INK, C_INK2 = "#898781", "#c3c2b7", "#e1e0d9", "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
MODEL_COLORS = {"scikit-learn LinearRegression": C_SK, "PyTorch manual loop": C_MAN, "PyTorch nn.Module + SGD": C_MOD}
MODELS = list(MODEL_COLORS)

# --------------------------------------------------------------------------
# Load artifacts (inference only)
# --------------------------------------------------------------------------
fb: P.FeatureBuilder = joblib.load(ART / "feature_builder.joblib")
sk_model = joblib.load(ART / "sklearn_linear_regression.joblib")
manual = torch.load(ART / "torch_manual.pt")
module_model = P.LinearRegressionModel(len(fb.feature_names_))
module_model.load_state_dict(torch.load(ART / "torch_module.pt"))
module_model.eval()
metrics = json.loads((ART / "metrics.json").read_text())
history = json.loads((ART / "training_history.json").read_text())
CFG = metrics["config"]
listings = pd.read_csv(ART / "listings_model.csv.gz", dtype={"ASSETID": str})
polygons = P.load_neighbourhoods()

is_test = (listings["SPLIT"] == "test").to_numpy()
train_rows = listings[~is_test]
test_rows = listings[is_test].reset_index(drop=True)


def predict_all(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """log(EUR/m2) predictions of the three saved models."""
    X = fb.transform(df)
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return {
            "scikit-learn LinearRegression": sk_model.predict(X),
            "PyTorch manual loop": (Xt @ manual["w"] + manual["b"]).numpy(),
            "PyTorch nn.Module + SGD": module_model(Xt).numpy(),
        }


PRED_TEST = predict_all(test_rows)
Y_TEST = test_rows["LOG_UNITPRICE"].to_numpy()
NB_MEDIAN = train_rows.groupby("NEIGHBOURHOOD")["LOG_UNITPRICE"].median()
GLOBAL_MEDIAN = float(train_rows["LOG_UNITPRICE"].median())
BASELINES = {
    "baseline: global median": np.full(len(test_rows), GLOBAL_MEDIAN),
    "baseline: neighbourhood median": test_rows["NEIGHBOURHOOD"].map(NB_MEDIAN).fillna(GLOBAL_MEDIAN).to_numpy(),
}
WEIGHTS = {
    "scikit-learn LinearRegression": sk_model.coef_,
    "PyTorch manual loop": manual["w"].numpy(),
    "PyTorch nn.Module + SGD": module_model.linear.weight.detach().numpy().ravel(),
}
UNSCALED = fb.unscaled_frame(listings)
UNSCALED = UNSCALED[[c for c in UNSCALED.columns if not c.startswith("nbhd=")]]


def style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=C_INK), x=0.01),
        template="plotly_white", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12, color=C_INK2),
        margin=dict(l=60, r=20, t=60, b=50), hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=C_GRID, linecolor=C_MUTED, zeroline=False)
    fig.update_yaxes(gridcolor=C_GRID, linecolor=C_MUTED, zeroline=False)
    return fig


def eur(v: float) -> str:
    return f"{v:,.0f} €".replace(",", " ")


# --------------------------------------------------------------------------
# Tab 1 — model comparison
# --------------------------------------------------------------------------
def metrics_table() -> pd.DataFrame:
    rows = []
    for name, pred in {**BASELINES, **PRED_TEST}.items():
        m = P.regression_metrics(Y_TEST, pred)
        rows.append({"model": name, "RMSE (log)": m["RMSE_log"], "R² (log)": m["R2_log"],
                     "MdAPE %": m["MdAPE_%"], "within ±10% (PPE10) %": m["PPE10_%"],
                     "within ±20% %": m["PPE20_%"], "median bias %": m["median_bias_%"]})
    return pd.DataFrame(rows).round(3)


def agreement_table() -> pd.DataFrame:
    ref = PRED_TEST["scikit-learn LinearRegression"]
    rows = []
    for name in MODELS[1:]:
        gap = 100 * np.abs(np.exp(PRED_TEST[name] - ref) - 1)
        rows.append({"model vs scikit-learn": name,
                     "max |Δ weight|": float(np.abs(WEIGHTS[name] - WEIGHTS["scikit-learn LinearRegression"]).max()),
                     "median |Δ predicted price| %": float(np.median(gap)),
                     "max |Δ predicted price| %": float(gap.max())})
    return pd.DataFrame(rows).round(5)


def pred_vs_actual(scale: str) -> go.Figure:
    to_unit = {"log(€/m²)": lambda v: v, "€/m²": np.exp,
               "price (€)": lambda v: np.exp(v) * test_rows["CONSTRUCTEDAREA"].to_numpy()}[scale]
    actual = to_unit(Y_TEST)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=MODELS)
    lo, hi = np.quantile(actual, [0.001, 0.999])
    for i, name in enumerate(MODELS, start=1):
        pred = to_unit(PRED_TEST[name])
        m = P.regression_metrics(Y_TEST, PRED_TEST[name])
        fig.add_trace(go.Scattergl(
            x=actual, y=pred, mode="markers", name=name, showlegend=False,
            marker=dict(size=4, color=MODEL_COLORS[name], opacity=0.25),
            hovertemplate="actual %{x:,.3s}<br>predicted %{y:,.3s}<extra>" + name + "</extra>"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=C_INK, width=1, dash="dash")), row=1, col=i)
        fig.add_annotation(text=f"MdAPE {m['MdAPE_%']:.2f}% · R² {m['R2_log']:.3f}", showarrow=False,
                           xref=f"x{'' if i == 1 else i} domain", yref=f"y{'' if i == 1 else i} domain",
                           x=0.02, y=0.98, xanchor="left", font=dict(size=11, color=C_INK2))
        fig.update_xaxes(title_text=f"actual {scale}", range=[lo, hi], row=1, col=i)
    fig.update_yaxes(title_text=f"predicted {scale}", range=[lo, hi], row=1, col=1)
    return style(fig, f"Predicted vs actual on the Q4 test quarter ({len(test_rows):,} listings) — dashed line = perfect", 430)


def error_distribution() -> go.Figure:
    fig = go.Figure()
    bins = np.linspace(-60, 60, 121)
    outside = 0
    for name in MODELS:
        err = 100 * (np.exp(PRED_TEST[name] - Y_TEST) - 1)
        outside = int((np.abs(err) > 60).sum())
        counts, edges = np.histogram(err, bins=bins)  # errors beyond +-60% are left out of the plot
        fig.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=counts, mode="lines", name=name,
                                 line=dict(color=MODEL_COLORS[name], width=2, shape="hvh"),
                                 hovertemplate="error %{x:.0f}%: %{y} flats<extra>" + name + "</extra>"))
    fig.add_vline(x=0, line=dict(color=C_INK, width=1))
    fig.add_annotation(text=f"{outside:,} listings with errors beyond ±60% not shown", showarrow=False,
                       xref="paper", yref="paper", x=1, y=0.95, xanchor="right", font=dict(size=11, color=C_INK2))
    fig.update_xaxes(title_text="valuation error: predicted / asking − 1 (%)")
    fig.update_yaxes(title_text="listings")
    return style(fig, "Valuation error distribution — the three curves lie on top of each other", 360)


def coefficient_plot(include_nbhd: bool, top_n: int) -> go.Figure:
    coef = pd.DataFrame(WEIGHTS, index=fb.feature_names_)
    if not include_nbhd:
        coef = coef[~coef.index.str.startswith("nbhd=")]
    coef = coef.reindex(coef["scikit-learn LinearRegression"].abs().sort_values(ascending=False).index).head(int(top_n))[::-1]
    fig = go.Figure()
    for name in MODELS:
        fig.add_trace(go.Bar(y=coef.index, x=coef[name], orientation="h", name=name,
                             marker=dict(color=MODEL_COLORS[name], cornerradius=3),
                             hovertemplate="%{y}: %{x:.4f}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(title_text="coefficient: change in log(€/m²) per +1 standard deviation of the feature")
    return style(fig, "Coefficients of the three implementations (standardised features)", 220 + 26 * len(coef))


def convergence_plot() -> go.Figure:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        f"Manual loop — full-batch GD, lr={CFG['lr_manual']}", f"nn.Module — mini-batch SGD, lr={CFG['lr_module']}"))
    for col, (key, color) in enumerate([("manual", C_MAN), ("module", C_MOD)], start=1):
        h = history[key]
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["dist_to_sklearn"], mode="lines", showlegend=False,
                                 line=dict(color=color, width=2),
                                 hovertemplate="epoch %{x}: %{y:.2e}<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="epoch", row=1, col=col)
        fig.update_yaxes(type="log", exponentformat="power", title_text="max |w − w_sklearn|" if col == 1 else None,
                         row=1, col=col)
    return style(fig, "Distance of each PyTorch model's weights to the exact scikit-learn solution during training", 360)


def config_markdown() -> str:
    c = CFG
    return (
        f"**Universe** {c['universe']} · **target** {c['target']} · **split** model selection {c['split']['selection']}, "
        f"final {c['split']['final']} · **{c['n_train']:,}** train / **{c['n_test']:,}** test flats · "
        f"**{c['n_features']}** features  \n"
        f"**Model** {c['model']} · **manual loop** full-batch gradient descent, lr = {c['lr_manual']}, "
        f"{c['epochs_manual']:,} epochs · **nn.Module** mini-batch SGD, lr = {c['lr_module']}, batch {c['batch_size']}, "
        f"{c['epochs_module']} epochs"
    )


# --------------------------------------------------------------------------
# Tab 2 — distributions
# --------------------------------------------------------------------------
TARGET_CHOICES = {"€/m² (target, before log)": np.exp(listings["LOG_UNITPRICE"]),
                  "log(€/m²) (model target)": listings["LOG_UNITPRICE"],
                  "asking price (€)": listings["PRICE"], "m²": listings["CONSTRUCTEDAREA"]}
FEATURE_CHOICES = list(TARGET_CHOICES) + list(UNSCALED.columns)


def _series(name: str) -> pd.Series:
    return (TARGET_CHOICES[name] if name in TARGET_CHOICES else UNSCALED[name]).reset_index(drop=True)


def distribution_plot(name: str) -> go.Figure:
    s = _series(name)
    values = np.sort(s.dropna().unique())
    discrete = len(values) <= 15 and np.allclose(values, np.round(values))  # dummies, floors, quarters...
    fig = go.Figure()
    for split, label, color in [("train", "train (Q1–Q3)", C_SK), ("test", "test (Q4)", C_MAN)]:
        v = s[(listings["SPLIT"] == split).to_numpy()]
        if discrete:
            share = v.value_counts(normalize=True).reindex(values).fillna(0)
            fig.add_trace(go.Bar(x=[f"{x:g}" for x in values], y=100 * share.values, name=label,
                                 marker=dict(color=color, cornerradius=3),
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>"))
        else:
            lo, hi = s.quantile([0.005, 0.995])
            fig.add_trace(go.Histogram(x=v.clip(lo, hi), name=label, histnorm="percent", nbinsx=60, opacity=0.55,
                                       marker=dict(color=color),
                                       hovertemplate="%{x}: %{y:.2f}%<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group" if discrete else "overlay")
    fig.update_xaxes(title_text=name)
    fig.update_yaxes(title_text="% of listings in split")
    return style(fig, f"Distribution of {name}: training vs test quarter", 380)


def relationship_plot(name: str) -> go.Figure:
    s = _series(name)[~is_test]
    y = np.exp(train_rows["LOG_UNITPRICE"].reset_index(drop=True))
    values = np.sort(s.dropna().unique())
    if len(values) <= 15 and np.allclose(values, np.round(values)):  # discrete: one point per value
        g = y.groupby(s.to_numpy()).agg(["median", "count"])
        g = g[g["count"] >= 30]
        x, med, n = [f"{k:g}" for k in g.index], g["median"], g["count"]
    else:
        q = pd.qcut(s.rank(method="first"), 12, labels=False)
        g = pd.DataFrame({"x": s.to_numpy(), "y": y.to_numpy(), "q": q.to_numpy()}).groupby("q").agg(
            x=("x", "median"), med=("y", "median"), n=("y", "size"))
        x, med, n = g["x"], g["med"], g["n"]
    fig = go.Figure(go.Scatter(x=x, y=med, mode="lines+markers", line=dict(color=C_SK, width=2),
                               marker=dict(size=9, color=C_SK, line=dict(color=SURFACE, width=2)),
                               customdata=n, hovertemplate="%{x}: median %{y:,.0f} €/m² (%{customdata} flats)<extra></extra>"))
    fig.update_xaxes(title_text=f"{name} (bin median)")
    fig.update_yaxes(title_text="median €/m²")
    return style(fig, f"Median €/m² against {name} — training data, 12 equal-count bins", 380)


def neighbourhood_plot(districts: list[str]) -> go.Figure:
    d = train_rows[train_rows["DISTRICT"].isin(districts or list(P.ALMENDRA_CENTRAL.values()))]
    g = d.groupby(["NEIGHBOURHOOD", "DISTRICT"])["LOG_UNITPRICE"].agg(["median", "size"]).reset_index()
    g["eur"] = np.exp(g["median"])
    g = g.sort_values("eur")
    fig = go.Figure(go.Bar(y=g["NEIGHBOURHOOD"], x=g["eur"], orientation="h",
                           marker=dict(color=C_SK, cornerradius=3), customdata=g[["DISTRICT", "size"]].to_numpy(dtype=object),
                           hovertemplate="%{y} (%{customdata[0]}): %{x:,.0f} €/m², %{customdata[1]} flats<extra></extra>"))
    fig.update_xaxes(title_text="median €/m² (Q1–Q3)")
    return style(fig, "Median €/m² by neighbourhood", 200 + 18 * len(g))


def drift_plot() -> go.Figure:
    g = listings.groupby("QUARTER")["LOG_UNITPRICE"].agg(["median", "size"])
    colors = [C_SK, C_SK, C_SK, C_MAN]
    fig = go.Figure(go.Scatter(x=g.index, y=np.exp(g["median"]), mode="lines+markers+text",
                               text=[f"{v:,.0f}" for v in np.exp(g["median"])], textposition="top center",
                               line=dict(color=C_MUTED, width=2), marker=dict(size=11, color=colors),
                               customdata=g["size"], hovertemplate="%{x}: %{y:,.0f} €/m² (%{customdata:,} flats)<extra></extra>"))
    fig.update_yaxes(title_text="median €/m²")
    return style(fig, "Median €/m² by quarter — the drift that forces a time-based split (orange = test)", 330)


# --------------------------------------------------------------------------
# Tab 3 — bargain radar (the business decision)
# --------------------------------------------------------------------------
CENTRAL_POLYS = [f for f in polygons["features"]
                 if f["properties"]["LOCATIONID"].split("-")[-2] in P.ALMENDRA_CENTRAL]
POLY_X, POLY_Y = [], []
for f in CENTRAL_POLYS:
    for poly in f["geometry"]["coordinates"]:
        ring = np.asarray(poly[0])
        POLY_X += list(ring[:, 0]) + [None]
        POLY_Y += list(ring[:, 1]) + [None]
CONDITION = np.select([test_rows["BUILTTYPEID_1"] == 1, test_rows["BUILTTYPEID_2"] == 1],
                      ["new build", "to renovate"], "good condition")


def screen(model: str, threshold: float, cap: float, districts: list[str]):
    """discount per Q4 listing, and the masks: flagged (threshold <= d < cap), implausible (d >= cap), in scope."""
    pred = PRED_TEST[model]
    disc = P.discount_vs_fair_value(Y_TEST, pred)
    scope = test_rows["DISTRICT"].isin(districts or []).to_numpy()
    fair = np.exp(pred) * test_rows["CONSTRUCTEDAREA"].to_numpy()
    flag = scope & (disc >= threshold) & (disc < cap)
    implausible = scope & (disc >= cap)
    return disc, flag, implausible, scope, fair


def kpi_html(model, threshold, cap, cost, districts) -> str:
    disc, flag, implausible, scope, fair = screen(model, threshold, cap, districts)
    n, n_scope = int(flag.sum()), int(scope.sum())
    gross = float((fair - test_rows["PRICE"].to_numpy())[flag].sum())
    med = float(np.median(disc[flag])) if n else 0.0
    tiles = [("Flagged for a visit (Q4)", f"{n:,}", f"{100 * n / max(n_scope, 1):.1f}% of {n_scope:,} listings in scope"),
             ("Due-diligence cost", eur(n * cost), f"{n:,} visits × {eur(cost)}"),
             ("Apparent discount, gross", f"{gross / 1e6:,.1f} M€", "sum of fair value − asking, before diligence"),
             ("Median discount of flags", f"{100 * med:.1f}%", f"band {100 * threshold:.0f}%–{100 * cap:.0f}% · model MdAPE ≈ {CFG['validation_mdape']:.0f}%"),
             ("Set aside: implausible", f"{int(implausible.sum()):,}", f"≥ {100 * cap:.0f}% below fair value: check listing type")]
    cells = "".join(
        f"<div class='kpi'><div class='kpi-label'>{a}</div><div class='kpi-value'>{b}</div>"
        f"<div class='kpi-sub'>{c}</div></div>" for a, b, c in tiles)
    return f"<div class='kpi-row'>{cells}</div>"


def _hover(mask, disc, fair):
    t = test_rows
    return pd.DataFrame({"n": t["NEIGHBOURHOOD"], "m2": t["CONSTRUCTEDAREA"], "p": t["PRICE"], "f": fair,
                         "d": 100 * disc, "c": CONDITION})[mask].to_numpy(dtype=object)


HOVER = ("<b>%{customdata[0]}</b><br>%{customdata[1]} m² · %{customdata[5]}<br>asking %{customdata[2]:,.0f} €"
         "<br>fair value %{customdata[3]:,.0f} €<br>discount %{customdata[4]:.1f}%<extra></extra>")


def map_plot(model, threshold, cap, districts) -> go.Figure:
    disc, flag, implausible, scope, fair = screen(model, threshold, cap, districts)
    t = test_rows
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=POLY_X, y=POLY_Y, mode="lines", line=dict(color=C_MUTED, width=0.8),
                             hoverinfo="skip", showlegend=False))
    rest = scope & ~flag & ~implausible
    fig.add_trace(go.Scattergl(x=t.loc[rest, "LONGITUDE"], y=t.loc[rest, "LATITUDE"], mode="markers",
                               name="not flagged", marker=dict(size=4, color=C_MUTED, opacity=0.45), hoverinfo="skip"))
    fig.add_trace(go.Scattergl(
        x=t.loc[implausible, "LONGITUDE"], y=t.loc[implausible, "LATITUDE"], mode="markers",
        name=f"implausible (≥ {100 * cap:.0f}%): check listing type",
        marker=dict(size=7, color=C_INK2, symbol="x-thin", line=dict(color=C_INK2, width=1.5)),
        customdata=_hover(implausible, disc, fair), hovertemplate=HOVER))
    fig.add_trace(go.Scattergl(
        x=t.loc[flag, "LONGITUDE"], y=t.loc[flag, "LATITUDE"], mode="markers",
        name=f"flagged ({100 * threshold:.0f}–{100 * cap:.0f}% below fair value)",
        marker=dict(size=8, color=C_MAN, opacity=0.85, line=dict(color=SURFACE, width=1)),
        customdata=_hover(flag, disc, fair), hovertemplate=HOVER))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1 / np.cos(np.radians(40.42)))
    fig = style(fig, f"Q4 listings screened by the {model} model", 640)
    fig.update_layout(margin=dict(l=10, r=10, t=80, b=10))
    return fig


def workload_plot(model, threshold, cap, cost, districts) -> go.Figure:
    disc, _, _, scope, _ = screen(model, threshold, cap, districts)
    grid = np.round(np.arange(0.0, 0.51, 0.01), 2)
    counts = np.array([(scope & (disc >= g) & (disc < cap)).sum() for g in grid])
    fig = go.Figure(go.Scatter(x=100 * grid, y=counts, mode="lines", line=dict(color=C_SK, width=2),
                               customdata=counts * cost,
                               hovertemplate="threshold %{x:.0f}%: %{y:,} visits, cost %{customdata:,.0f} €<extra></extra>"))
    cur = int((scope & (disc >= threshold) & (disc < cap)).sum())
    fig.add_trace(go.Scatter(x=[100 * threshold], y=[cur], mode="markers+text", text=[f"{cur:,} visits · {eur(cur * cost)}"],
                             textposition="top right", marker=dict(size=11, color=C_MAN, line=dict(color=SURFACE, width=2)),
                             hoverinfo="skip"))
    fig.add_vline(x=100 * CFG["bargain_threshold"], line=dict(color=C_BASE, width=1, dash="dot"),
                  annotation_text="default d (90th pct. of validation)", annotation_position="top")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="threshold d: minimum discount vs fair value (%)", range=[0, 50])
    fig.update_yaxes(title_text="listings to visit in Q4")
    return style(fig, "Workload: visits implied by each threshold", 380)


def top_table(model, threshold, cap, districts) -> pd.DataFrame:
    disc, flag, _, _, fair = screen(model, threshold, cap, districts)
    t = test_rows.assign(discount=100 * disc, fair=fair, condition=CONDITION)[flag]
    t = t.sort_values("discount", ascending=False).head(25)
    return pd.DataFrame({
        "neighbourhood": t["NEIGHBOURHOOD"], "district": t["DISTRICT"], "m²": t["CONSTRUCTEDAREA"],
        "rooms": t["ROOMNUMBER"], "floor": t["FLOORCLEAN"], "lift": t["HASLIFT"].map({1: "yes", 0: "no"}),
        "condition": t["condition"], "asking €": t["PRICE"].round(0), "fair value €": t["fair"].round(-3),
        "discount %": t["discount"].round(1)})


def radar(model, threshold, cap, cost, districts):
    cap = max(cap, threshold + 0.01)
    return (kpi_html(model, threshold, cap, cost, districts), map_plot(model, threshold, cap, districts),
            workload_plot(model, threshold, cap, cost, districts), top_table(model, threshold, cap, districts))


# --------------------------------------------------------------------------
# Tab 4 — value one flat
# --------------------------------------------------------------------------
NB_DEFAULTS = train_rows.groupby("NEIGHBOURHOOD")[
    ["DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_CASTELLANA", "CADMAXBUILDINGFLOOR",
     "CADDWELLINGCOUNT", "CADASTRALQUALITYID", "LATITUDE", "LONGITUDE"]].median()
NB_DISTRICT = train_rows.groupby("NEIGHBOURHOOD")["DISTRICT"].first()
NEIGHBOURHOODS = sorted(NB_DEFAULTS.index)


def value_flat(nbhd, m2, rooms, baths, floor, lift, exterior, condition, year, terrace, ac, parking, doorman, pool, asking):
    d = NB_DEFAULTS.loc[nbhd]
    row = pd.DataFrame([{
        "PERIOD": 201812, "NEIGHBOURHOOD": nbhd, "CONSTRUCTEDAREA": float(m2), "ROOMNUMBER": int(rooms),
        "BATHNUMBER": int(baths), "FLOORCLEAN": float(floor), "HASLIFT": int(lift),
        "FLATLOCATIONID": 1 if exterior else 2, "BUILTTYPEID_1": int(condition == "new build"),
        "BUILTTYPEID_2": int(condition == "to renovate"), "CADCONSTRUCTIONYEAR": int(year),
        "HASTERRACE": int(terrace), "HASAIRCONDITIONING": int(ac), "ISPARKINGSPACEINCLUDEDINPRICE": int(parking),
        "HASDOORMAN": int(doorman), "HASSWIMMINGPOOL": int(pool), "ISSTUDIO": int(rooms == 0), "ISDUPLEX": 0,
        "ISINTOPFLOOR": 0, "HASSOUTHORIENTATION": 0, "HASBOXROOM": 0, "HASWARDROBE": 1, "HASGARDEN": 0,
        "AMENITYID": 3, **d.to_dict()}])
    preds = predict_all(row)
    comps = float(np.exp(NB_MEDIAN.get(nbhd, GLOBAL_MEDIAN)))
    lines = [f"<tr><td><span class='dot' style='background:{MODEL_COLORS[k]}'></span>{k}</td>"
             f"<td>{np.exp(v[0]):,.0f} €/m²</td><td><b>{eur(np.exp(v[0]) * m2)}</b></td></tr>"
             for k, v in preds.items()]
    lines.append(f"<tr><td><span class='dot' style='background:{C_BASE}'></span>comps baseline (neighbourhood median)</td>"
                 f"<td>{comps:,.0f} €/m²</td><td>{eur(comps * m2)}</td></tr>")
    html = (f"<div class='valuation'><p class='muted'>{nbhd} ({NB_DISTRICT[nbhd]}) · valued as a Q4-2018 listing · "
            f"building height, units, cadastral quality and distances set to the neighbourhood median</p>"
            f"<table><tr><th>model</th><th>fair €/m²</th><th>fair price</th></tr>{''.join(lines)}</table>")
    if asking and asking > 0:
        fair = float(np.exp(preds["scikit-learn LinearRegression"][0]) * m2)
        disc = 1 - asking / fair
        verdict = ("implausibly cheap: check whether it is bare ownership, an auction or an occupied flat"
                   if disc >= CFG["bargain_cap"]
                   else "<b>flagged</b>: worth a visit" if disc >= CFG["bargain_threshold"]
                   else "not flagged: within what the model's error can explain" if disc > -CFG["bargain_threshold"]
                   else "asking well above fair value")
        html += (f"<p>Asking {eur(asking)} is <b>{100 * disc:+.1f}%</b> vs the scikit-learn fair value "
                 f"(screen threshold {100 * CFG['bargain_threshold']:.0f}%): {verdict}.</p>")
    return html + "</div>"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
CSS = """
.kpi-row {display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px;}
.kpi {background:#fcfcfb; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:12px 14px;}
.kpi-label {font-size:12px; color:#52514e;}
.kpi-value {font-size:26px; font-weight:650; color:#0b0b0b; margin:2px 0; font-variant-numeric:tabular-nums;}
.kpi-sub {font-size:11.5px; color:#898781;}
.valuation table {width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums;}
.valuation td, .valuation th {padding:6px 8px; border-bottom:1px solid #e1e0d9; text-align:left;}
.valuation .muted {color:#898781; font-size:12px;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px;}
"""

best = metrics_table().set_index("model")
HEADER = f"""
# Fair value of central-Madrid flats
A buy-to-let investor screens new for-sale listings in Madrid's 7 central districts. One model — linear regression on
**log(€/m²)** — trained three ways on Q1–Q3 2018 and tested on Q4. On the test quarter it values the median listing within
**{best.loc['scikit-learn LinearRegression', 'MdAPE %']:.1f}%** of its asking price, against **{best.loc['baseline: neighbourhood median', 'MdAPE %']:.1f}%**
for the "neighbourhood median €/m²" rule of thumb. Every number on this page is computed from the saved models; nothing is retrained.
"""

with gr.Blocks(title="Madrid fair-value dashboard") as demo:
    gr.Markdown(HEADER)
    with gr.Tabs():
        with gr.Tab("1 · Model comparison"):
            gr.Markdown(config_markdown())
            gr.Dataframe(metrics_table(), label="Q4 test metrics, computed live from the saved models", interactive=False)
            scale = gr.Radio(["log(€/m²)", "€/m²", "price (€)"], value="log(€/m²)", label="Scale")
            pva = gr.Plot(pred_vs_actual("log(€/m²)"))
            scale.change(pred_vs_actual, scale, pva)
            gr.Plot(error_distribution())
            gr.Dataframe(agreement_table(), label="Agreement with scikit-learn (same objective, same data)", interactive=False)
            with gr.Row():
                include_nbhd = gr.Checkbox(False, label="Include neighbourhood dummies")
                top_n = gr.Slider(10, 60, value=25, step=5, label="Number of coefficients")
            coef_plot = gr.Plot(coefficient_plot(False, 25))
            include_nbhd.change(coefficient_plot, [include_nbhd, top_n], coef_plot)
            top_n.release(coefficient_plot, [include_nbhd, top_n], coef_plot)
            gr.Plot(convergence_plot())

        with gr.Tab("2 · Distributions"):
            gr.Plot(drift_plot())
            feature = gr.Dropdown(FEATURE_CHOICES, value=FEATURE_CHOICES[0], label="Feature or target")
            with gr.Row():
                dist = gr.Plot(distribution_plot(FEATURE_CHOICES[0]))
                rel = gr.Plot(relationship_plot(FEATURE_CHOICES[0]))
            feature.change(lambda f: (distribution_plot(f), relationship_plot(f)), feature, [dist, rel])
            nb_districts = gr.CheckboxGroup(list(P.ALMENDRA_CENTRAL.values()), value=list(P.ALMENDRA_CENTRAL.values()),
                                            label="Districts")
            nb_plot = gr.Plot(neighbourhood_plot(list(P.ALMENDRA_CENTRAL.values())))
            nb_districts.change(neighbourhood_plot, nb_districts, nb_plot)

        with gr.Tab("3 · Bargain radar"):
            gr.Markdown(
                "A Q4 listing is **flagged** when it asks at least *d*% less than the model's fair value. The default *d* "
                "is the 90th percentile of discounts on the validation quarter, about 1.7 times the model's typical error. "
                "Discounts above the **plausibility cap** (99th percentile) are set aside: asking 80% below comparable "
                "flats signals bare ownership, an auction, an occupied flat or a typo, not a bargain. "
                "Each flag costs a visit and a preliminary appraisal. A flag means *cheap given what the ad says*, so it "
                "is a visit list, not a buy list.")
            with gr.Row():
                model_in = gr.Radio(MODELS, value=MODELS[0], label="Model")
                cost_in = gr.Slider(0, 2000, value=CFG["cost_per_visit_eur"], step=50, label="Cost per visit (€)")
            with gr.Row():
                thr_in = gr.Slider(0.0, 0.5, value=round(CFG["bargain_threshold"], 2), step=0.01,
                                   label="Threshold d: minimum discount vs fair value")
                cap_in = gr.Slider(0.2, 0.9, value=round(CFG["bargain_cap"], 2), step=0.01,
                                   label="Plausibility cap: larger discounts are set aside (bare ownership, auctions, typos)")
            dist_in = gr.CheckboxGroup(list(P.ALMENDRA_CENTRAL.values()), value=list(P.ALMENDRA_CENTRAL.values()),
                                       label="Districts in scope")
            init = radar(MODELS[0], round(CFG["bargain_threshold"], 2), round(CFG["bargain_cap"], 2),
                         CFG["cost_per_visit_eur"], list(P.ALMENDRA_CENTRAL.values()))
            kpis = gr.HTML(init[0])
            with gr.Row():
                map_out = gr.Plot(init[1], scale=3)
                with gr.Column(scale=2):
                    work_out = gr.Plot(init[2])
            table_out = gr.Dataframe(init[3], label="25 largest discounts among flagged listings", interactive=False)
            inputs, outputs = [model_in, thr_in, cap_in, cost_in, dist_in], [kpis, map_out, work_out, table_out]
            model_in.change(radar, inputs, outputs)
            dist_in.change(radar, inputs, outputs)
            for slider in (thr_in, cap_in, cost_in):
                slider.release(radar, inputs, outputs)

        with gr.Tab("4 · Value a flat"):
            gr.Markdown("Describe a flat and get its fair asking price from the three saved models. "
                        "Add an asking price to see where it falls against the bargain screen.")
            with gr.Row():
                with gr.Column():
                    v_nb = gr.Dropdown(NEIGHBOURHOODS, value="Trafalgar", label="Neighbourhood")
                    with gr.Row():
                        v_m2 = gr.Number(85, label="m²", minimum=20, maximum=400)
                        v_rooms = gr.Number(2, label="bedrooms", minimum=0, maximum=10, precision=0)
                        v_baths = gr.Number(1, label="bathrooms", minimum=1, maximum=6, precision=0)
                    with gr.Row():
                        v_floor = gr.Number(3, label="floor (0 = ground, -1 = semi-basement)", minimum=-1, maximum=20, precision=0)
                        v_year = gr.Number(1960, label="construction year", minimum=1700, maximum=2018, precision=0)
                    v_cond = gr.Radio(["good condition", "to renovate", "new build"], value="good condition", label="Condition")
                    with gr.Row():
                        v_lift = gr.Checkbox(True, label="lift")
                        v_ext = gr.Checkbox(True, label="exterior")
                        v_terr = gr.Checkbox(False, label="terrace")
                        v_ac = gr.Checkbox(False, label="air conditioning")
                    with gr.Row():
                        v_park = gr.Checkbox(False, label="parking included in price")
                        v_door = gr.Checkbox(False, label="doorman")
                        v_pool = gr.Checkbox(False, label="pool")
                    v_ask = gr.Number(None, label="asking price (€, optional)")
                    v_btn = gr.Button("Value this flat", variant="primary")
                with gr.Column():
                    v_inputs = [v_nb, v_m2, v_rooms, v_baths, v_floor, v_lift, v_ext, v_cond, v_year,
                                v_terr, v_ac, v_park, v_door, v_pool, v_ask]
                    v_out = gr.HTML(value_flat("Trafalgar", 85, 2, 1, 3, True, True, "good condition", 1960,
                                               False, False, False, False, False, None))
            v_btn.click(value_flat, v_inputs, v_out)


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(primary_hue="blue"), css=CSS)
