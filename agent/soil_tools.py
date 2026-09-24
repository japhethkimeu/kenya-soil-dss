"""
Step 5a - Soil tools for the smolagents CodeAgent.

The tools are the agent's only access to the soil data. Each one reads the
outputs of steps 3 and 4 and returns plain Python dicts:

  find_place(name)                          place name -> coordinates (OpenStreetMap)
  get_soil_at_location(lat, lon)            values, classes, confidence, AOA,
                                            interpretation for every property
  summarize_area(lat, lon, radius_km)       class distribution around a point
  get_class_definitions(property_name)      threshold table (classes, ranges,
                                            interpretation, source)

Inputs read (paths relative to the repo root, or set SOIL_DSS_ROOT):
  outputs/maps/<prop>_0_30cm.tif            q05, q25, q50, q75, q95, pi90_width, di, aoa
  outputs/indicators/<prop>_classes.tif     class_q50, class_q25, class_q75, confidence, aoa
  config/indicator_thresholds.csv

The logic lives in plain functions (_point_query, _area_summary, ...) so it can
later be reused as an MCP server. The @tool wrappers only add docstrings for the LLM.

Usage:
  from agent.soil_tools import TOOLS, get_soil_at_location
  get_soil_at_location(lat=0.28, lon=34.75)      # call a tool directly
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import Window
from smolagents import tool

ROOT = Path(os.environ.get("SOIL_DSS_ROOT", "."))
MAPS = ROOT / "outputs" / "maps"
IND = ROOT / "outputs" / "indicators"
THRESHOLDS = ROOT / "config" / "indicator_thresholds.csv"
DEPTH = "0_30cm"
NODATA = -9999

# Display settings per property. Add a line here when a new property is mapped.
PROPERTIES = {
    "ph": {"label": "Soil pH (water)", "method": "H2O", "decimals": 1},
    "soc_gkg": {"label": "Soil organic carbon", "method": "any", "decimals": 1},
}
ALIASES = {
    "ph": "ph", "soil ph": "ph", "acidity": "ph", "ph_h2o": "ph",
    "soc": "soc_gkg", "soc_gkg": "soc_gkg", "organic carbon": "soc_gkg",
    "soil organic carbon": "soc_gkg", "carbon": "soc_gkg", "organic matter": "soc_gkg",
}
CONFIDENCE = {1: "high", 2: "medium", 3: "low"}


# ------------------------------------------------------------------ helpers
def _thresholds(prop):
    t = pd.read_csv(THRESHOLDS)
    t = t[(t.property == prop) & (t.method == PROPERTIES[prop]["method"])]
    return t.sort_values("class_id").set_index("class_id")


def _range_text(row):
    if pd.isna(row["min"]):
        return f"< {row['max']:g} {row.unit}"
    if pd.isna(row["max"]):
        return f">= {row['min']:g} {row.unit}"
    return f"{row['min']:g} to <{row['max']:g} {row.unit}"


def _available():
    return [p for p in PROPERTIES
            if (MAPS / f"{p}_{DEPTH}.tif").exists() and (IND / f"{p}_classes.tif").exists()]


def _bands(ds):
    return {name: i + 1 for i, name in enumerate(ds.descriptions)}


def _to_rowcol(ds, lat, lon):
    x, y = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(lon, lat)
    row, col = ds.index(x, y)
    inside = 0 <= row < ds.height and 0 <= col < ds.width
    return row, col, x, y, inside


def _clean(obj):
    """Convert numpy scalars/arrays inside nested dicts/lists to plain Python types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _resolve(name):
    key = str(name).strip().lower()
    return ALIASES.get(key, key if key in PROPERTIES else None)


# ------------------------------------------------------------------ core logic
def _point_query(lat, lon):
    result = {"location": {"lat": round(lat, 5), "lon": round(lon, 5)},
              "depth": "0-30 cm", "properties": {}}
    for prop in _available():
        cfg, thr = PROPERTIES[prop], _thresholds(prop)
        with rasterio.open(MAPS / f"{prop}_{DEPTH}.tif") as m, \
             rasterio.open(IND / f"{prop}_classes.tif") as c:
            row, col, _, _, inside = _to_rowcol(m, lat, lon)
            if not inside:
                result["properties"][prop] = {"label": cfg["label"],
                                              "status": "outside the mapped area"}
                continue
            win = Window(col, row, 1, 1)
            mv = {n: float(m.read(b, window=win)[0, 0]) for n, b in _bands(m).items()}
            cv = {n: int(c.read(b, window=win)[0, 0]) for n, b in _bands(c).items()}
        if mv["q50"] == NODATA or cv["class_q50"] == 0:
            result["properties"][prop] = {"label": cfg["label"],
                                          "status": "no data at this location"}
            continue

        d = cfg["decimals"]
        cls = thr.loc[cv["class_q50"]]
        lo_id, hi_id = sorted((cv["class_q25"], cv["class_q75"]))
        possible = [thr.loc[i, "class"] for i in range(lo_id, hi_id + 1) if i in thr.index]
        conf = CONFIDENCE.get(cv["confidence"], "unknown")
        in_aoa = bool(mv["aoa"] == 1)
        others = [c for c in possible if c != cls["class"]]
        if conf == "high" or not others:
            cls_text = f"{cls['class']} (high confidence)"
        elif conf == "medium":
            cls_text = f"{cls['class']}, possibly {others[0].lower()} (medium confidence)"
        else:
            cls_text = (f"{cls['class']}; could be anywhere between {possible[0].lower()} "
                        f"and {possible[-1].lower()} (low confidence)")
        aoa_text = ("inside the area well represented by the sampled sites" if in_aoa else
                    "OUTSIDE the area well represented by the sampled sites, "
                    "so this estimate is less reliable")
        summary = (f"{cfg['label']}: {mv['q50']:.{d}f} {cls.unit} - {cls_text}; "
                   f"likely range {mv['q25']:.{d}f}-{mv['q75']:.{d}f} {cls.unit}; {aoa_text}.")
        result["properties"][prop] = {
            "summary": summary,
            "label": cfg["label"],
            "unit": cls.unit,
            "predicted_value": round(mv["q50"], d),
            "likely_range_50pct": [round(mv["q25"], d), round(mv["q75"], d)],
            "range_90pct": [round(mv["q05"], d), round(mv["q95"], d)],
            "class": cls["class"],
            "class_range": _range_text(cls),
            "possible_classes": possible,
            "confidence": conf,
            "inside_area_of_applicability": in_aoa,
            "interpretation": cls.interpretation,
            "explanation": cls.explanation,
            "threshold_source": cls.source,
        }
    if not result["properties"]:
        result["error"] = "No soil maps found. Run steps 3 and 4 first."
    return result


def _area_summary(lat, lon, radius_km):
    radius_km = float(min(max(radius_km, 0.5), 50))
    result = {"center": {"lat": round(lat, 5), "lon": round(lon, 5)},
              "radius_km": radius_km, "depth": "0-30 cm", "properties": {}}
    for prop in _available():
        cfg, thr = PROPERTIES[prop], _thresholds(prop)
        with rasterio.open(MAPS / f"{prop}_{DEPTH}.tif") as m, \
             rasterio.open(IND / f"{prop}_classes.tif") as c:
            row, col, x, y, _ = _to_rowcol(m, lat, lon)
            px = abs(m.transform.a)
            r = int(np.ceil(radius_km * 1000 / px))
            r0, c0 = max(row - r, 0), max(col - r, 0)
            r1, c1 = min(row + r + 1, m.height), min(col + r + 1, m.width)
            if r0 >= r1 or c0 >= c1:
                result["properties"][prop] = {"label": cfg["label"],
                                              "status": "outside the mapped area"}
                continue
            win = Window(c0, r0, c1 - c0, r1 - r0)
            mb, cb = _bands(m), _bands(c)
            q50 = m.read(mb["q50"], window=win)
            cls = c.read(cb["class_q50"], window=win)
            conf = c.read(cb["confidence"], window=win)
            aoa = c.read(cb["aoa"], window=win)
            rows, cols = np.mgrid[r0:r1, c0:c1]
            xs, ys = rasterio.transform.xy(m.transform, rows.ravel(), cols.ravel())
            dist = np.hypot(np.array(xs) - x, np.array(ys) - y).reshape(rows.shape)
        valid = (dist <= radius_km * 1000) & (cls > 0) & (q50 != NODATA)
        n = int(valid.sum())
        if n == 0:
            result["properties"][prop] = {"label": cfg["label"],
                                          "status": "no mapped data in this area"}
            continue
        d = cfg["decimals"]
        dist_cls = []
        for cid, cnt in zip(*np.unique(cls[valid], return_counts=True)):
            dist_cls.append({"class": thr.loc[cid, "class"],
                             "class_range": _range_text(thr.loc[cid]),
                             "percent_of_area": round(100 * float(cnt) / n, 1)})
        dist_cls.sort(key=lambda k: -k["percent_of_area"])
        conf_pct = {v: round(100 * float((conf[valid] == k).mean()), 1)
                    for k, v in CONFIDENCE.items()}
        aoa_pct = round(100 * float((aoa[valid] == 1).mean()), 1)
        dom = thr[thr["class"] == dist_cls[0]["class"]].iloc[0]
        dist_text = ", ".join(f"{c['class'].lower()} {c['percent_of_area']:.0f}%"
                              for c in dist_cls)
        conf_text = ", ".join(f"{v:.0f}% {k}" for k, v in conf_pct.items() if v >= 0.5)
        summary = (f"{cfg['label']} within {radius_km:g} km: mean "
                   f"{float(q50[valid].mean()):.{d}f} {thr.unit.iloc[0]}; area by class: "
                   f"{dist_text}; confidence: {conf_text} of the area; "
                   f"{aoa_pct:.0f}% of the area is well represented by the sampled sites.")
        result["properties"][prop] = {
            "summary": summary,
            "label": cfg["label"],
            "unit": thr.unit.iloc[0],
            "mapped_area_km2": round(n * px * px / 1e6, 1),
            "mean_value": round(float(q50[valid].mean()), d),
            "min_max_value": [round(float(q50[valid].min()), d),
                              round(float(q50[valid].max()), d)],
            "dominant_class": dist_cls[0]["class"],
            "dominant_class_interpretation": dom.interpretation,
            "dominant_class_explanation": dom.explanation,
            "class_distribution": dist_cls,
            "confidence_percent": conf_pct,
            "percent_inside_area_of_applicability": aoa_pct,
            "threshold_source": dom.source,
        }
    if not result["properties"]:
        result["error"] = "No soil maps found. Run steps 3 and 4 first."
    return result


def _class_table(property_name):
    props = _available() if str(property_name).lower() == "all" else [_resolve(property_name)]
    if props == [None]:
        return {"error": f"Unknown property '{property_name}'. "
                         f"Available: {', '.join(PROPERTIES)} or 'all'."}
    out = {}
    for p in props:
        thr = _thresholds(p)
        out[p] = {"label": PROPERTIES[p]["label"],
                  "source": thr.source.iloc[0],
                  "classes": [{"class": r["class"], "range": _range_text(r),
                               "interpretation": r.interpretation,
                               "explanation": r.explanation}
                              for _, r in thr.iterrows()]}
    return out


def _geocode(name):
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": name, "countrycodes": "ke", "format": "json", "limit": 3},
            headers={"User-Agent": "kenya-soil-dss-prototype"}, timeout=15)
        resp.raise_for_status()
        hits = resp.json()
    except Exception as e:
        return {"error": f"Geocoding failed: {e}"}
    if not hits:
        return {"error": f"No place called '{name}' found in Kenya."}
    return {"query": name,
            "matches": [{"name": h["display_name"], "lat": round(float(h["lat"]), 5),
                         "lon": round(float(h["lon"]), 5), "type": h.get("type")}
                        for h in hits]}


# ------------------------------------------------------------------ smolagents tools
@tool
def find_place(name: str) -> dict:
    """Find the coordinates of a place in Kenya (town, village, ward, market) by name.
    Use this before the soil tools when the user gives a place name instead of coordinates.

    Returns a dict (NOT a list):
        {"query": <str>,
         "matches": [{"name": <str>, "lat": <float>, "lon": <float>, "type": <str>}, ...]}
    Up to 3 matches; use result["matches"][0] unless the user's wording fits another
    match better. On failure returns {"error": <str>} instead.

    Args:
        name: Place name, e.g. "Kitale", "Malava", "Kakamega town".
    """
    return _clean(_geocode(name))


@tool
def get_soil_at_location(lat: float, lon: float) -> dict:
    """Get the mapped topsoil (0-30 cm) conditions at ONE location (a single farm or
    point) for all available properties: soil pH ("ph") and soil organic carbon ("soc_gkg").

    Returns a dict:
        {"location": {"lat": <float>, "lon": <float>}, "depth": "0-30 cm",
         "properties": {
            "ph": {"summary": <ready-to-use sentence: value, class, confidence, AOA>,
                   "label": <str>, "unit": <str>,
                   "predicted_value": <float>,
                   "likely_range_50pct": [<float>, <float>],
                   "range_90pct": [<float>, <float>],
                   "class": <str>, "class_range": <str>,
                   "possible_classes": [<str>, ...],
                   "confidence": "high" | "medium" | "low",
                   "inside_area_of_applicability": <bool>,
                   "interpretation": <management meaning of the class>,
                   "explanation": <why the class matters for crops>,
                   "threshold_source": <str>},
            "soc_gkg": {...same keys...}}}
    Use the "summary" field to report each property. If a property has a "status"
    key instead (e.g. "outside the mapped area"), there is no data there.

    Args:
        lat: Latitude in decimal degrees (WGS84), e.g. 0.28.
        lon: Longitude in decimal degrees (WGS84), e.g. 34.75.
    """
    return _clean(_point_query(lat, lon))


@tool
def summarize_area(lat: float, lon: float, radius_km: float) -> dict:
    """Summarize topsoil (0-30 cm) conditions within a radius around a point, for all
    available properties. Use this for questions about an AREA: a town and its
    surroundings, a ward, a village, or "near <place>".

    Returns a dict:
        {"center": {"lat": <float>, "lon": <float>}, "radius_km": <float>, "depth": "0-30 cm",
         "properties": {
            "ph": {"summary": <ready-to-use sentence: mean, % of area per class,
                               confidence mix, % inside area of applicability>,
                   "label": <str>, "unit": <str>,
                   "mapped_area_km2": <float>, "mean_value": <float>,
                   "min_max_value": [<float>, <float>],
                   "dominant_class": <str>,
                   "dominant_class_interpretation": <management meaning>,
                   "dominant_class_explanation": <why it matters for crops>,
                   "class_distribution": [{"class": <str>, "class_range": <str>,
                                           "percent_of_area": <float 0-100>}, ...],
                   "confidence_percent": {"high": <float 0-100>, "medium": <float 0-100>,
                                          "low": <float 0-100>},
                   "percent_inside_area_of_applicability": <float 0-100>,
                   "threshold_source": <str>},
            "soc_gkg": {...same keys...}}}
    All percentages are ALREADY 0-100: never multiply them by 100.
    Area results have no "class", "possible_classes" or "confidence" keys.
    Use the "summary" field to report each property.

    Args:
        lat: Latitude of the area centre in decimal degrees (WGS84).
        lon: Longitude of the area centre in decimal degrees (WGS84).
        radius_km: Radius in kilometres (0.5 to 50). Use 3 km for a town or village
            and its surroundings unless the user gives a radius.
    """
    return _clean(_area_summary(lat, lon, radius_km))


@tool
def get_class_definitions(property_name: str) -> dict:
    """Get the class thresholds used to interpret a soil property: each class with its
    value range, management interpretation, explanation of why it matters for crops,
    and the source of the thresholds. Use this for questions about what a class or
    value means, or why it matters.

    Returns a dict keyed by property:
        {"ph": {"label": <str>, "source": <str>,
                "classes": [{"class": <str>, "range": <str>,
                             "interpretation": <str>, "explanation": <str>}, ...]}}
    Find a class by looping over result["ph"]["classes"].

    Args:
        property_name: 'ph', 'soc' (soil organic carbon), or 'all' for every property.
    """
    return _clean(_class_table(property_name))


TOOLS = [find_place, get_soil_at_location, summarize_area, get_class_definitions]
