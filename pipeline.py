"""
Shared code for the two stages: train.ipynb (training) and app.py (dashboard).

Why a separate module? The dashboard must turn a listing into model inputs in
*exactly* the same way as training did, using the numbers learned from the
training data (medians for missing values, the list of neighbourhoods, the
mean and standard deviation of each feature). If that code were copied into
two places, the two copies could drift apart and the app would feed the
models slightly different inputs ("train/serve skew"). One module, used by
both stages, rules that out.

Contents
    1. loading the committed data
    2. clean()          -> which listings we keep, and why
    3. row_features()   -> the engineered features
    4. FeatureBuilder   -> learns medians / neighbourhoods / scaling on TRAIN only
    5. LinearRegressionModel -> the nn.Module used by the standard PyTorch workflow
    6. metrics
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

# idealista publishes one snapshot per quarter of 2018 (PERIOD = yyyymm)
QUARTER_LABEL = {201803: "Q1", 201806: "Q2", 201809: "Q3", 201812: "Q4"}
QUARTER_INDEX = {201803: 0, 201806: 1, 201809: 2, 201812: 3}

# "Almendra central": Madrid's 7 central districts, the investor's market.
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

MAX_AREA_M2 = 400     # bigger = whole buildings / mansions, outside the mandate
TYPICAL_AREA_M2 = 90  # the median flat; the squared size term is measured from here


# ==========================================================================
# 1. Loading
# ==========================================================================
def load_listings() -> pd.DataFrame:
    """All 94,815 Madrid for-sale ads of 2018 (built by data/prepare_dataset.py)."""
    return pd.read_csv(DATA_DIR / "madrid_sale_2018.csv.gz", dtype={"ASSETID": str})


def load_neighbourhoods() -> dict:
    """The 135 idealista neighbourhood polygons (GeoJSON)."""
    return json.loads((DATA_DIR / "madrid_barrios.geojson").read_text())


def assign_neighbourhood(df: pd.DataFrame, polygons: dict) -> pd.DataFrame:
    """The data has coordinates but no neighbourhood name, so we look up which
    neighbourhood polygon each flat's point falls inside ("point in polygon").

    This is safe despite the anonymisation: the publisher moved every flat
    30-60 m but always kept it inside its own neighbourhood.
    """
    points = df[["LONGITUDE", "LATITUDE"]].to_numpy()
    name = np.full(len(df), None, dtype=object)
    loc_id = np.full(len(df), None, dtype=object)
    for feat in polygons["features"]:
        inside = np.zeros(len(df), dtype=bool)
        for polygon in feat["geometry"]["coordinates"]:
            for ring in polygon:  # a polygon can have holes: each ring flips inside/outside
                inside ^= MplPath(np.asarray(ring)).contains_points(points)
        new = inside & pd.isna(name)
        name[new] = feat["properties"]["LOCATIONNAME"]
        loc_id[new] = feat["properties"]["LOCATIONID"]
    out = df.copy()
    out["NEIGHBOURHOOD"] = name
    out["DISTRICT_ID"] = pd.Series(loc_id, index=df.index).str.split("-").str[-2]
    out["DISTRICT"] = out["DISTRICT_ID"].map(ALMENDRA_CENTRAL)
    return out


# ==========================================================================
# 2. Cleaning
# Only row filters that do not learn anything from the data, so they can be
# applied before the train/test split without leaking information.
# ==========================================================================
def clean(df: pd.DataFrame, polygons: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (modelling dataset, log of how many rows each rule removed)."""
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
    # THE TARGET: log of the asking price per square metre
    cur["LOG_UNITPRICE"] = np.log(cur["PRICE"] / cur["CONSTRUCTEDAREA"])
    return cur, pd.DataFrame(log)


# ==========================================================================
# 3. Feature engineering
# ==========================================================================
# Control model for the comparison: the published columns as they come.
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

# Construction eras. Price vs. age is not a straight line: old pre-war
# buildings and recent ones are worth more than the 1960-79 blocks. One dummy
# per era lets the model give each era its own premium. 1960-79 is the
# reference era (no dummy), so each coefficient reads "vs. a 1960-79 building".
ERAS = [
    ("era_pre1900", -np.inf, 1900),
    ("era_1900_1939", 1900, 1940),
    ("era_1940_1959", 1940, 1960),
    ("era_1980_1999", 1980, 2000),
    ("era_2000_plus", 2000, np.inf),
]


def row_features(df: pd.DataFrame, imputes: dict) -> pd.DataFrame:
    """Engineered features computed row by row.
    `imputes` holds medians learned on the training set (to fill missing values)."""
    f = pd.DataFrame(index=df.index)
    area = df["CONSTRUCTEDAREA"].astype(float)
    rooms = df["ROOMNUMBER"].astype(float)
    floor = df["FLOORCLEAN"].astype(float)
    lift = df["HASLIFT"].astype(float)
    year = df["CADCONSTRUCTIONYEAR"].astype(float)
    quality = df["CADASTRALQUALITYID"].astype(float).replace(0, np.nan)  # 0 = not recorded

    # --- trend -----------------------------------------------------------
    # Prices rise during 2018. A number 0,1,2,3 lets a straight line carry the
    # trend into Q4. A "Q4" dummy would be useless: Q4 is never in training.
    f["t_quarter"] = df["PERIOD"].map(QUARTER_INDEX).astype(float)

    # --- location (the neighbourhood dummies are added in FeatureBuilder) --
    # Logs: the first few hundred metres matter much more than the next ones.
    # +0.05 km (50 m) avoids log(0) for a flat right on top of the metro.
    f["log_dist_castellana"] = np.log(df["DISTANCE_TO_CASTELLANA"] + 0.05)
    f["log_dist_sol"] = np.log(df["DISTANCE_TO_CITY_CENTER"] + 0.05)
    f["log_dist_metro"] = np.log(df["DISTANCE_TO_METRO"] + 0.05)

    # --- building ----------------------------------------------------------
    for name, lo, hi in ERAS:
        f[name] = ((year >= lo) & (year < hi)).astype(float)
    f["cadastral_quality"] = quality.fillna(imputes["cadastral_quality"])  # 1 = best ... 9 = worst
    f["new_build"] = df["BUILTTYPEID_1"].astype(float)
    f["needs_renovation"] = df["BUILTTYPEID_2"].astype(float)
    f["log_building_floors"] = np.log(df["CADMAXBUILDINGFLOOR"].clip(lower=1).astype(float))
    f["log_building_units"] = np.log1p(df["CADDWELLINGCOUNT"].astype(float))
    f["doorman"] = df["HASDOORMAN"].astype(float)

    # --- floor x lift ------------------------------------------------------
    # A high floor is worth more only if there is a lift. floors_without_lift
    # counts the floors you climb on foot: an INTERACTION (floor x no lift).
    floor_filled = floor.fillna(imputes["floor"])
    f["floor"] = floor_filled.clip(lower=0)
    f["floor_unknown"] = floor.isna().astype(float)  # missing indicator: the advertiser hid it
    f["semi_basement"] = (floor_filled < 0).astype(float)
    f["ground_floor"] = (floor_filled == 0).astype(float)
    f["lift"] = lift
    f["floors_without_lift"] = floor_filled.clip(lower=0) * (1 - lift)
    f["top_floor"] = df["ISINTOPFLOOR"].astype(float)
    f["exterior"] = (df["FLATLOCATIONID"] == 1).astype(float)
    f["exterior_unknown"] = df["FLATLOCATIONID"].isna().astype(float)
    f["south_facing"] = df["HASSOUTHORIENTATION"].astype(float)

    # --- size & extras -------------------------------------------------------
    # EUR/m2 falls as flats get bigger, then rises again for very large ones:
    # a U-shape. A straight line cannot bend, so we add the square of log(m2).
    # It is measured from a typical 90 m2 flat so that log_m2 and its square
    # are not near-copies of each other (that would slow gradient descent).
    f["log_m2"] = np.log(area)
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
    # a parking space included in the asking price raises EUR/m2 without more flat
    f["parking_included"] = df["ISPARKINGSPACEINCLUDEDINPRICE"].astype(float)
    f["no_kitchen_furniture"] = (df["AMENITYID"] == 1).astype(float)
    f["furnished"] = (df["AMENITYID"] == 3).astype(float)
    return f


BLOCK_COLUMNS = {
    "trend": ["t_quarter"],
    "location": ["log_dist_castellana", "log_dist_sol", "log_dist_metro"],  # + one dummy per neighbourhood
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
    """Turns cleaned listings into the model's input matrix X.

    fit_transform(train) LEARNS three things from the training rows only:
        - medians used to fill missing values,
        - the list of neighbourhoods (one dummy each, except a reference one),
        - the mean and standard deviation of every feature (for standardising).
    transform(other) then APPLIES those same numbers to validation, test or a
    flat typed into the dashboard. Learning them on test data would leak
    information from the future into the model.

    blocks: which engineered blocks to use, or ("raw",) for the control model
            that uses the published columns as they come.
    """

    def __init__(self, blocks=tuple(ENGINEERED_BLOCKS)):
        self.blocks = tuple(blocks)

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
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
            if block == "location":
                # one 0/1 column per neighbourhood; the first one (alphabetical) is
                # the reference and gets no column, otherwise the dummies would add
                # up to the intercept (the "dummy variable trap")
                for nb in self.neighbourhoods_[1:]:
                    rows[f"nbhd={nb}"] = (df["NEIGHBOURHOOD"] == nb).astype(float).to_numpy()
                    cols.append(f"nbhd={nb}")
        return rows[cols]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        quality = df["CADASTRALQUALITYID"].replace(0, np.nan)
        self.imputes_ = {
            "floor": float(df["FLOORCLEAN"].median()),
            "cadastral_quality": float(quality.median()),
        }
        self.neighbourhoods_ = sorted(df["NEIGHBOURHOOD"].unique())
        frame = self._frame(df)
        self.feature_names_ = list(frame.columns)
        X = frame.to_numpy(dtype=float)
        # Standardise: every feature gets mean 0 and std 1 on the training set.
        # Gradient descent needs features on comparable scales (m2 in the
        # hundreds vs. dummies in 0/1), otherwise one learning rate cannot suit all.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (self.unscaled_frame(df).to_numpy(dtype=float) - self.mean_) / self.std_

    def unscaled_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features in their natural units (before standardising), for plots."""
        return self._frame(df)


# ==========================================================================
# 5. The model for the standard PyTorch workflow (the app reloads it)
# ==========================================================================
class LinearRegressionModel(nn.Module):
    """prediction = X @ w + b. One nn.Linear layer and no activation function,
    so it is exactly a linear regression."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


# ==========================================================================
# 6. Metrics
# The target is log(EUR/m2). Going back to euros, the m2 cancel out, so a %
# error on EUR/m2 is the same % error on the price.
# ==========================================================================
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
        "MdAPE_%": float(100 * np.median(ape)),        # median % error: "typical" miss
        "PPE10_%": float(100 * np.mean(ape <= 0.10)),  # % of flats valued within +-10%
        "PPE20_%": float(100 * np.mean(ape <= 0.20)),
        "median_bias_%": float(100 * (np.exp(np.median(-resid)) - 1)),  # > 0: overvalues
    }


def discount_vs_fair_value(y_true_log, y_pred_log) -> np.ndarray:
    """1 - asking / fair value. 0.20 = 'asks 20% less than the model says it is worth'."""
    return 1 - np.exp(np.asarray(y_true_log, dtype=float) - np.asarray(y_pred_log, dtype=float))
