"""
Step 4 - Indicators: classify prediction rasters using config/indicator_thresholds.csv

For every property with a prediction raster (outputs/maps/<prop>_0_30cm.tif,
bands q05, q25, q50, q75, q95, pi90_width, di, aoa) this writes:

  outputs/indicators/<prop>_classes.tif   uint8, nodata 0, 5 bands:
      1 class_q50     class of the median prediction (the headline class)
      2 class_q25     class of the lower quartile
      3 class_q75     class of the upper quartile
      4 confidence    from the 50% interval (q25..q75):
                      1 = high (one class), 2 = medium (two adjacent classes),
                      3 = low (three or more classes)
      5 aoa           1 = inside area of applicability, 2 = outside
  outputs/indicators/<prop>_class_summary.csv   area (km2, %) per class,
                                                all pixels and inside the AOA
  outputs/indicators/<prop>_classes.png         quick-look maps

Thresholds are read from the config, so changing classes or adding a property
needs no code change - only rows in the CSV (and METHOD below if relevant).
Run from the repo root:  python scripts/04_indicators.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

THRESHOLDS = "config/indicator_thresholds.csv"
MAPS = Path("outputs/maps")
OUT = Path("outputs/indicators")
DEPTH = "0_30cm"
NODATA_IN = -9999
# which method's thresholds to use per property (matches the 'method' column)
METHOD = {"ph": "H2O", "soc_gkg": "any"}
CMAPS = {"ph": "RdYlBu", "soc_gkg": "YlOrBr"}

OUT.mkdir(parents=True, exist_ok=True)
thr = pd.read_csv(THRESHOLDS)


def classify(values, table):
    """Class id (1..n) for each value; 0 where value is nodata."""
    lows = table["min"].fillna(-np.inf).to_numpy()
    ids = table["class_id"].to_numpy()
    idx = np.searchsorted(lows, values, side="right") - 1
    out = ids[np.clip(idx, 0, len(ids) - 1)].astype("uint8")
    out[~np.isfinite(values)] = 0
    return out


for prop in thr.property.unique():
    src_path = MAPS / f"{prop}_{DEPTH}.tif"
    if not src_path.exists():
        print(f"[skip] {prop}: no prediction raster at {src_path}")
        continue
    table = thr[(thr.property == prop) &
                (thr.method == METHOD.get(prop, "any"))].sort_values("class_id")
    if table.empty:
        print(f"[skip] {prop}: no thresholds for method {METHOD.get(prop)}")
        continue

    with rasterio.open(src_path) as src:
        band = {name: i + 1 for i, name in enumerate(src.descriptions)}
        q = src.read([band["q25"], band["q50"], band["q75"]]).astype("float64")
        aoa = src.read(band["aoa"])
        prof = src.profile
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    q[q == NODATA_IN] = np.nan

    c25, c50, c75 = (classify(q[i], table) for i in (0, 1, 2))
    valid = c50 > 0
    span = np.abs(c75.astype(int) - c25.astype(int))
    conf = np.where(span == 0, 1, np.where(span == 1, 2, 3)).astype("uint8")
    conf[~valid] = 0
    aoa_b = np.where(aoa == 1, 1, 2).astype("uint8")
    aoa_b[~valid] = 0

    # ---- raster
    prof.update(count=5, dtype="uint8", nodata=0, compress="deflate")
    out_tif = OUT / f"{prop}_classes.tif"
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(np.stack([c50, c25, c75, conf, aoa_b]))
        dst.descriptions = ("class_q50", "class_q25", "class_q75", "confidence", "aoa")

    # ---- area summary
    rows = []
    for _, r in table.iterrows():
        m = c50 == r.class_id
        rows.append({"class_id": r.class_id, "class": r["class"],
                     "range": (f"< {r['max']:g}" if pd.isna(r["min"]) else
                               f">= {r['min']:g}" if pd.isna(r["max"]) else
                               f"{r['min']:g} - <{r['max']:g}") + f" {r.unit}",
                     "area_km2": m.sum() * px_km2,
                     "area_km2_in_aoa": (m & (aoa_b == 1)).sum() * px_km2})
    summ = pd.DataFrame(rows)
    summ["pct"] = 100 * summ.area_km2 / summ.area_km2.sum()
    summ["pct_in_aoa"] = 100 * summ.area_km2_in_aoa / max(summ.area_km2_in_aoa.sum(), 1e-9)
    conf_pct = {k: 100 * (conf == v).sum() / valid.sum()
                for k, v in [("high", 1), ("medium", 2), ("low", 3)]}
    summ.round(1).to_csv(OUT / f"{prop}_class_summary.csv", index=False)

    print(f"\n=== {prop}  ->  {out_tif}")
    print(summ.round(1).to_string(index=False))
    print("confidence (% of area): " +
          ", ".join(f"{k} {v:.0f}%" for k, v in conf_pct.items()))

    # ---- quick-look
    n = len(table)
    colors = plt.get_cmap(CMAPS.get(prop, "viridis"))(np.linspace(0.05, 0.95, n))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    img = np.ma.masked_equal(c50, 0)
    axes[0].imshow(img, cmap=ListedColormap(colors), vmin=0.5, vmax=n + 0.5,
                   extent=extent, interpolation="nearest")
    present = set(np.unique(c50[valid]))
    axes[0].legend(handles=[Patch(color=colors[i], label=f"{r['class']}")
                            for i, (_, r) in enumerate(table.iterrows())
                            if r.class_id in present],
                   loc="lower left", fontsize=8)
    axes[0].set_title(f"{prop} class (median prediction)")
    cc = ["#1a9850", "#fee08b", "#d73027"]
    axes[1].imshow(np.ma.masked_equal(conf, 0), cmap=ListedColormap(cc),
                   vmin=0.5, vmax=3.5, extent=extent, interpolation="nearest")
    axes[1].legend(handles=[Patch(color=c, label=l) for c, l in
                            zip(cc, ["High - one class", "Medium - two classes",
                                     "Low - three or more"])],
                   loc="lower left", fontsize=8)
    axes[1].set_title("Class confidence (50% interval, q25-q75)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{prop} indicator, {DEPTH.replace('_', '-')}  |  thresholds: "
                 f"{table.source.iloc[0].split(' - ')[0]}", fontsize=10)
    fig.savefig(OUT / f"{prop}_classes.png", dpi=130)
    plt.close(fig)
