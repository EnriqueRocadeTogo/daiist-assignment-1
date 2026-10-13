# /// script
# requires-python = ">=3.12"
# dependencies = ["rdata>=0.11", "pandas>=2.2", "numpy>=2.0"]
# ///
"""
One-off script that builds the committed dataset from its public source.

Source: the `idealista18` R package (Rey-Blanco, Arbués & López, 2024,
https://github.com/paezha/idealista18), which publishes every for-sale
listing on idealista.com in Madrid, Barcelona and Valencia during 2018,
enriched with cadastral data. We only use Madrid.

It is NOT part of the graded pipeline (`main.py train` reads the CSV this
script writes), so it declares its own dependencies above and runs with:

    uv run data/prepare_dataset.py

Outputs (both committed):
    data/madrid_sale_2018.csv.gz   one row per listing ad, 41 columns, as published
    data/madrid_barrios.geojson    the 135 idealista neighbourhood polygons
"""

import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

warnings.filterwarnings("ignore")  # rdata warns about R classes it maps loosely

BASE_URL = "https://github.com/paezha/idealista18/raw/master/data/"
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def download(name: str) -> Path:
    RAW.mkdir(exist_ok=True)
    path = RAW / name
    if not path.exists():
        urllib.request.urlretrieve(BASE_URL + name, path)
    return path


def to_geojson_geometry(geom) -> dict:
    """sf geometries come out of rdata as nested lists of (n, 2) arrays."""

    def depth(x):
        return 0 if isinstance(x, np.ndarray) else 1 + depth(x[0])

    if depth(geom) == 1:  # POLYGON: list of rings
        return {"type": "Polygon", "coordinates": [r.tolist() for r in geom]}
    return {  # MULTIPOLYGON: list of polygons, each a list of rings
        "type": "MultiPolygon",
        "coordinates": [[r.tolist() for r in poly] for poly in geom],
    }


def main() -> None:
    sale = rdata.read_rda(download("Madrid_Sale.rda"))["Madrid_Sale"]
    sale = sale.drop(columns="geometry")  # same info as LONGITUDE / LATITUDE
    sale["ASSETID"] = sale["ASSETID"].astype(str)
    out_csv = HERE / "madrid_sale_2018.csv.gz"
    sale.to_csv(out_csv, index=False, compression="gzip")
    print(f"{out_csv.name}: {sale.shape[0]:,} rows x {sale.shape[1]} cols")

    polys = rdata.read_rda(download("Madrid_Polygons.rda"), default_encoding="utf8")[
        "Madrid_Polygons"
    ]
    features = [
        {
            "type": "Feature",
            "properties": {
                "LOCATIONID": str(row.LOCATIONID),
                "LOCATIONNAME": str(row.LOCATIONNAME),
            },
            "geometry": to_geojson_geometry(row.geometry),
        }
        for row in polys.itertuples()
    ]
    out_geo = HERE / "madrid_barrios.geojson"
    out_geo.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"{out_geo.name}: {len(features)} neighbourhoods")


if __name__ == "__main__":
    main()
