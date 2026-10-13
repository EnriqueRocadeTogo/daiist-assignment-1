"""
Shared code for the two stages: train.ipynb (training) and app.py (serving).

Why a separate module: the Gradio app must apply *exactly* the same cleaning
and feature transformations as training, with the statistics learned on the
training set (imputation values, scaler, neighbourhood list). Keeping that
logic in one place avoids train/serve skew. Everything that *learns* from
data lives in FeatureBuilder.fit_transform(); everything else is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path as MplPath
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# idealista publishes one snapshot per quarter of 2018
QUARTER_LABEL = {201803: "Q1", 201806: "Q2", 201809: "Q3", 201812: "Q4"}
QUARTER_INDEX = {201803: 0, 201806: 1, 201809: 2, 201812: 3}

# "Almendra central" (central almond): Madrid's 7 central districts.
# The district code is the second-to-last block of idealista's LOCATIONID.
ALMENDRA_CENTRAL = {
    "01": "Centro",
    "02": "Arganzuela",
    "03": "Retiro",
    "04": "Salamanca",
    "05": "Chamartín",
    "06": "Tetuán",
    "07": "Chamberí",
}

MAX_AREA_M2 = 400
TYPICAL_AREA_M2 = 90  # centre of the squared size term (the median flat is ~90 m2)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_listings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "madrid_sale_2018.csv.gz", dtype={"ASSETID": str})


def load_neighbourhoods() -> dict:
    return json.loads((DATA_DIR / "madrid_barrios.geojson").read_text())


def assign_neighbourhood(df: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """Point-in-polygon: which idealista neighbourhood each listing falls in.

    The dataset ships coordinates but no neighbourhood. Anonymisation moved
    every listing 30-60 m *within its own neighbourhood*, so the polygon
    lookup still returns the true one.
    """
    pts = df[["LONGITUDE", "LATITUDE"]].to_numpy()
    name = np.full(len(df), None, dtype=object)
    loc_id = np.full(len(df), None, dtype=object)
    for feat in polygons["features"]:
        inside = np.zeros(len(df), dtype=bool)
        for polygon in feat["geometry"]["coordinates"]:
            for ring in polygon:  # even-odd rule: a hole ring toggles back to outside
                inside ^= MplPath(np.asarray(ring)).contains_points(pts)
        new = inside & pd.isna(name)
        name[new] = feat["properties"]["LOCATIONNAME"]
        loc_id[new] = feat["properties"]["LOCATIONID"]
    out = df.copy()
    out["NEIGHBOURHOOD"] = name
    out["DISTRICT_ID"] = pd.Series(loc_id, index=df.index).str.split("-").str[-2]
    out["DISTRICT"] = out["DISTRICT_ID"].map(ALMENDRA_CENTRAL)
    return out


# --------------------------------------------------------------------------
# Cleaning: deterministic row filters, no statistics learned from the data,
# so it is safe to run before the train/test split.
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame, polygons: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the modelling dataset and a log of how many rows each rule removed."""
    log = []

    def step(name, new_df, reason):
        log.append({"step": name, "rows_before": len(cur), "rows_after": len(new_df), "reason": reason})
        return new_df

    cur = assign_neighbourhood(df, polygons)
    cur = step("Coordinates outside Madrid", cur[cur["NEIGHBOURHOOD"].notna()],
               "Fall in no neighbourhood polygon: broken coordinates (some are 400 km away)")
    cur = step("Universe: central almond", cur[cur["DISTRICT_ID"].isin(ALMENDRA_CENTRAL)],
               "The investor's target market: districts 01-07")
    cur = step("One flat = one row",
               cur.sort_values(["ASSETID", "PRICE"]).drop_duplicates("ASSETID", keep="first"),
               "Same flat advertised by several agencies at different prices in the same "
               "quarter: keep the cheapest, the price a buyer could actually get")
    cur = step("Zero bathrooms", cur[cur["BATHNUMBER"] > 0],
               "A flat for sale without a bathroom is a data-entry error")
    cur = step("Over 400 m2", cur[cur["CONSTRUCTEDAREA"] <= MAX_AREA_M2],
               "Whole buildings and mansions: a different product, outside a buy-to-let mandate")
    cur = cur.sort_values(["PERIOD", "ASSETID"]).reset_index(drop=True)
    cur["QUARTER"] = cur["PERIOD"].map(QUARTER_LABEL)
    cur["LOG_UNITPRICE"] = np.log(cur["PRICE"] / cur["CONSTRUCTEDAREA"])  # the target
    return cur, pd.DataFrame(log)


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
# Control model: the published columns as they come, minimally imputed.
RAW_COLUMNS = [
    "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "FLOORCLEAN", "HASLIFT",
    "HASTERRACE", "HASAIRCONDITIONING", "HASPARKINGSPACE", "HASNORTHORIENTATION",
    "HASSOUTHORIENTATION", "HASEASTORIENTATION", "HASWESTORIENTATION", "HASBOXROOM",
    "HASWARDROBE", "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN", "ISDUPLEX",
    "ISSTUDIO", "ISINTOPFLOOR", "FLATLOCATIONID", "AMENITYID", "CADCONSTRUCTIONYEAR",
    "CADMAXBUILDINGFLOOR", "CADDWELLINGCOUNT", "CADASTRALQUALITYID", "BUILTTYPEID_1",
    "BUILTTYPEID_2", "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO",
    "DISTANCE_TO_CASTELLANA", "LATITUDE", "LONGITUDE",
]

ENGINEERED_BLOCKS = ["trend", "location", "building", "floor", "size_extras"]

# Building eras: price vs. age is U-shaped (historic buildings and new builds
# carry a premium; the 1960-79 stock is the cheapest), which a single linear
# "year" term cannot capture. 1960-79 is the reference category.
ERAS = [
    ("era_pre1900", -np.inf, 1900),
    ("era_1900_1939", 1900, 1940),
    ("era_1940_1959", 1940, 1960),
    ("era_1980_1999", 1980, 2000),
    ("era_2000_plus", 2000, np.inf),
]


def row_features(df: pd.DataFrame, imputes: dict) -> pd.DataFrame:
    """Deterministic, row-wise features. `imputes` holds training-set medians."""
    f = pd.DataFrame(index=df.index)
    area = df["CONSTRUCTEDAREA"].astype(float)
    rooms = df["ROOMNUMBER"].astype(float)
    floor = df["FLOORCLEAN"].astype(float)
    lift = df["HASLIFT"].astype(float)
    year = df["CADCONSTRUCTIONYEAR"].astype(float)
    quality = df["CADASTRALQUALITYID"].astype(float).replace(0, np.nan)  # 0 = not recorded

    # trend: a *numeric* quarter index. A Q4 dummy would never be seen while
    # training on Q1-Q3, so only a numeric trend can extrapolate the price drift.
    f["t_quarter"] = df["PERIOD"].map(QUARTER_INDEX).astype(float)

    # location (the neighbourhood itself is added in FeatureBuilder): distances
    # in logs, because the first 500 m to the metro matter far more than going
    # from 2 to 2.5 km. The +50 m floor avoids log(0).
    f["log_dist_castellana"] = np.log(df["DISTANCE_TO_CASTELLANA"] + 0.05)
    f["log_dist_sol"] = np.log(df["DISTANCE_TO_CITY_CENTER"] + 0.05)
    f["log_dist_metro"] = np.log(df["DISTANCE_TO_METRO"] + 0.05)

    # building
    for name, lo, hi in ERAS:
        f[name] = ((year >= lo) & (year < hi)).astype(float)
    f["cadastral_quality"] = quality.fillna(imputes["cadastral_quality"])  # 1 best ... 9 worst
    f["new_build"] = df["BUILTTYPEID_1"].astype(float)
    f["needs_renovation"] = df["BUILTTYPEID_2"].astype(float)
    f["log_building_floors"] = np.log(df["CADMAXBUILDINGFLOOR"].clip(lower=1).astype(float))
    f["log_building_units"] = np.log1p(df["CADDWELLINGCOUNT"].astype(float))
    f["doorman"] = df["HASDOORMAN"].astype(float)

    # floor x lift: every floor you climb on foot is a penalty; with a lift it is not
    floor_filled = floor.fillna(imputes["floor"])
    f["floor"] = floor_filled.clip(lower=0)
    f["floor_unknown"] = floor.isna().astype(float)
    f["semi_basement"] = (floor_filled < 0).astype(float)
    f["ground_floor"] = (floor_filled == 0).astype(float)
    f["lift"] = lift
    f["floors_without_lift"] = floor_filled.clip(lower=0) * (1 - lift)
    f["top_floor"] = df["ISINTOPFLOOR"].astype(float)
    f["exterior"] = (df["FLATLOCATIONID"] == 1).astype(float)
    f["exterior_unknown"] = df["FLATLOCATIONID"].isna().astype(float)
    f["south_facing"] = df["HASSOUTHORIENTATION"].astype(float)

    # size & extras: EUR/m2 falls with size, so log(area) enters even though
    # the target is already per m2.
    f["log_m2"] = np.log(area)
    # U-shape: tiny studios and very large flats both carry a higher EUR/m2.
    # Centred at 90 m2 before squaring so it is not ~collinear with log_m2
    # (an uncentred square made X'X ~7x worse conditioned for gradient descent).
    f["log_m2_sq"] = (f["log_m2"] - np.log(TYPICAL_AREA_M2)) ** 2
    f["m2_per_room"] = area / rooms.clip(lower=1)
    f["baths_per_room"] = df["BATHNUMBER"].astype(float) / rooms.clip(lower=1)
    f["studio"] = df["ISSTUDIO"].astype(float)
    f["duplex"] = df["ISDUPLEX"].astype(float)
    f["terrace"] = df["HASTERRACE"].astype(float)
    f["air_conditioning"] = df["HASAIRCONDITIONING"].astype(float)
    f["storage_room"] = df["HASBOXROOM"].astype(float)
    f["wardrobes"] = df["HASWARDROBE"].astype(float)
    f["pool"] = df["HASSWIMMINGPOOL"].astype(float)
    f["garden"] = df["HASGARDEN"].astype(float)
    # a parking space included in the asking price inflates EUR/m2 without more flat
    f["parking_included"] = df["ISPARKINGSPACEINCLUDEDINPRICE"].astype(float)
    f["no_kitchen_furniture"] = (df["AMENITYID"] == 1).astype(float)
    f["furnished"] = (df["AMENITYID"] == 3).astype(float)
    return f


BLOCK_COLUMNS = {
    "trend": ["t_quarter"],
    "location": ["log_dist_castellana", "log_dist_sol", "log_dist_metro"],  # + neighbourhood
    "building": [e[0] for e in ERAS] + [
        "cadastral_quality", "new_build", "needs_renovation", "log_building_floors",
        "log_building_units", "doorman",
    ],
    "floor": [
        "floor", "floor_unknown", "semi_basement", "ground_floor", "lift",
        "floors_without_lift", "top_floor", "exterior", "exterior_unknown", "south_facing",
    ],
    "size_extras": [
        "log_m2", "log_m2_sq", "m2_per_room", "baths_per_room", "studio", "duplex",
        "terrace", "air_conditioning", "storage_room", "wardrobes", "pool", "garden",
        "parking_included", "no_kitchen_furniture", "furnished",
    ],
}


class FeatureBuilder:
    """Learns everything that needs training data, then transforms any split.

    blocks: engineered blocks to use, or ("raw",) for the no-feature-engineering
            control (the published columns as they come).
    neighbourhood_encoding: "onehot" (one dummy per neighbourhood) or "target"
            (smoothed, out-of-fold target encoding: one number per neighbourhood).
    """

    def __init__(self, blocks=tuple(ENGINEERED_BLOCKS), neighbourhood_encoding="onehot",
                 smoothing=20.0, n_folds=5, seed=42):
        self.blocks = tuple(blocks)
        self.neighbourhood_encoding = neighbourhood_encoding
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.seed = seed

    # ---- target encoding (only used when neighbourhood_encoding="target") --
    def _te_map(self, nbhd: pd.Series, y: pd.Series) -> tuple[dict, float]:
        """Shrinks each neighbourhood's mean target towards the global mean;
        `smoothing` acts like that many pseudo-observations at the global mean."""
        prior = float(y.mean())
        stats = y.groupby(nbhd).agg(["sum", "count"])
        enc = (stats["sum"] + self.smoothing * prior) / (stats["count"] + self.smoothing)
        return enc.to_dict(), prior

    def _te_out_of_fold(self, nbhd: pd.Series, y: pd.Series) -> np.ndarray:
        """Training rows get an encoding computed WITHOUT their own fold, so the
        model cannot learn to trust a feature that already contains its own target."""
        rng = np.random.default_rng(self.seed)
        fold = rng.integers(0, self.n_folds, size=len(y))
        out = np.empty(len(y))
        for k in range(self.n_folds):
            enc, prior = self._te_map(nbhd[fold != k], y[fold != k])
            out[fold == k] = nbhd[fold == k].map(enc).fillna(prior).to_numpy()
        return out

    # ---- design matrix ---------------------------------------------------
    def _frame(self, df: pd.DataFrame, nbhd_te: np.ndarray | None) -> pd.DataFrame:
        if self.blocks == ("raw",):
            raw = df[RAW_COLUMNS].astype(float)
            raw["FLOORCLEAN"] = raw["FLOORCLEAN"].fillna(self.imputes_["floor"])
            raw["FLATLOCATIONID"] = raw["FLATLOCATIONID"].fillna(1)
            raw["t_quarter"] = df["PERIOD"].map(QUARTER_INDEX).astype(float)
            return raw
        rows = row_features(df, self.imputes_)
        cols = []
        for block in self.blocks:
            cols += BLOCK_COLUMNS[block]
            if block != "location":
                continue
            if self.neighbourhood_encoding == "target":
                rows["neighbourhood_te"] = nbhd_te
                cols.append("neighbourhood_te")
            else:  # one-hot; the first neighbourhood (alphabetically) is the reference
                for nb in self.neighbourhoods_[1:]:
                    rows[f"nbhd={nb}"] = (df["NEIGHBOURHOOD"] == nb).astype(float).to_numpy()
                    cols.append(f"nbhd={nb}")
        return rows[cols]

    def fit_transform(self, df: pd.DataFrame, y) -> np.ndarray:
        y = pd.Series(np.asarray(y, dtype=float))
        nbhd = df["NEIGHBOURHOOD"].reset_index(drop=True)
        quality = df["CADASTRALQUALITYID"].replace(0, np.nan)
        self.imputes_ = {
            "floor": float(df["FLOORCLEAN"].median()),
            "cadastral_quality": float(quality.median()),
        }
        self.neighbourhoods_ = sorted(df["NEIGHBOURHOOD"].unique())
        self.te_map_, self.te_prior_ = self._te_map(nbhd, y)
        frame = self._frame(df, self._te_out_of_fold(nbhd, y))
        self.feature_names_ = list(frame.columns)
        X = frame.to_numpy(dtype=float)
        # standardise with TRAINING statistics only: gradient descent needs
        # comparable feature scales, and the ridge penalty treats all weights alike
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (self.unscaled_frame(df).to_numpy(dtype=float) - self.mean_) / self.std_

    def unscaled_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features before standardisation (human-readable units)."""
        te = df["NEIGHBOURHOOD"].map(self.te_map_).fillna(self.te_prior_).to_numpy(dtype=float)
        return self._frame(df, te)


# --------------------------------------------------------------------------
# Model used by the standard PyTorch workflow (and reloaded by the app)
# --------------------------------------------------------------------------
class LinearRegressionModel(nn.Module):
    """y_hat = X w + b. A single nn.Linear layer and no activation: linear regression."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


# --------------------------------------------------------------------------
# Metrics. The target is log(EUR/m2); m2 cancels when going back to price,
# so percentage errors on EUR/m2 are percentage errors on the price.
# --------------------------------------------------------------------------
def regression_metrics(y_true_log, y_pred_log) -> dict:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    resid = y_true_log - y_pred_log
    ape = np.abs(np.exp(-resid) - 1)  # |predicted price - asking price| / asking price
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true_log - y_true_log.mean()) ** 2)
    return {
        "RMSE_log": float(np.sqrt(np.mean(resid ** 2))),
        "MAE_log": float(np.mean(np.abs(resid))),
        "R2_log": float(1 - ss_res / ss_tot),
        "MdAPE_%": float(100 * np.median(ape)),
        "PPE10_%": float(100 * np.mean(ape <= 0.10)),
        "PPE20_%": float(100 * np.mean(ape <= 0.20)),
        "median_bias_%": float(100 * (np.exp(np.median(-resid)) - 1)),  # >0: overvalues
    }


def discount_vs_fair_value(y_true_log, y_pred_log) -> np.ndarray:
    """1 - asking / fair value. 0.15 = 'asking 15% below what the model says it is worth'."""
    return 1 - np.exp(np.asarray(y_true_log, dtype=float) - np.asarray(y_pred_log, dtype=float))
