"""
Step 1 - Build the pH and SOC point table for Kakamega and Trans Nzoia
from the final GloSIS-harmonized workbooks.

Output: data/points_ph_soc_kakamega_transnzoia.csv  (one row per site, 0-30 cm)
        outputs/points_ph_soc_map.png
Sites with more than one specimen are averaged.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("data/raw")   # GloSIS workbooks (run from the repo root)
FILES = {"Kakamega": "Kakamega_KCEP_AGRA_GloSIS_harmonized.xlsx",
         "Trans Nzoia": "TransNzoia_KCEP_GloSIS_harmonized.xlsx"}
PROPS = {"pH - Hydrogen potential": "ph", "Carbon (C) - total": "soc_gkg"}

rows = []
for county, fname in FILES.items():
    f = SRC / fname
    plot = pd.read_excel(f, "Plot Data", header=1)
    elem = pd.read_excel(f, "Element Data", header=1)
    spec = pd.read_excel(f, "Specimen Data")
    d = (spec[["profile_code", "element_code", *PROPS]]
         .merge(elem[["profile_code", "element_code", "upper_depth", "lower_depth"]],
                on=["profile_code", "element_code"])
         .merge(plot[["profile_code", "project_name", "longitude", "latitude"]],
                on="profile_code")
         .rename(columns=PROPS))
    d["county"] = county
    d["dataset"] = fname.replace("_GloSIS_harmonized.xlsx", "")
    rows.append(d)
spec_all = pd.concat(rows, ignore_index=True)

# one row per site: average specimens, keep count and within-site spread
g = spec_all.groupby(["dataset", "profile_code"])
sites = g.agg(county=("county", "first"), lon=("longitude", "first"),
              lat=("latitude", "first"), upper_depth=("upper_depth", "min"),
              lower_depth=("lower_depth", "max"), ph=("ph", "mean"),
              soc_gkg=("soc_gkg", "mean"), n_specimens=("element_code", "size")
              ).reset_index()
sites.insert(0, "site_id", sites.dataset + "_" + sites.profile_code)
sites[["ph", "soc_gkg"]] = sites[["ph", "soc_gkg"]].round(2)

Path("data").mkdir(exist_ok=True); Path("outputs").mkdir(exist_ok=True)
sites.to_csv("data/points_ph_soc_kakamega_transnzoia.csv", index=False)

# summary
multi = spec_all.groupby(["dataset", "profile_code"]).filter(lambda x: len(x) > 1)
spread = multi.groupby(["dataset", "profile_code"])[["ph", "soc_gkg"]].agg(lambda s: s.max() - s.min())
print("Sites per county:", sites.county.value_counts().to_dict())
print("Sites averaged from >1 specimen:", int((sites.n_specimens > 1).sum()))
print("Within-site range at averaged sites (median / max):")
print(spread.agg(["median", "max"]).round(2).to_string())
print("\nSummary by county:")
print(sites.groupby("county")[["ph", "soc_gkg"]]
      .describe(percentiles=[.05, .5, .95]).round(2).T.to_string())

# map
fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
for ax, (col, label, cmap) in zip(axes, [("ph", "pH", "RdYlBu"),
                                          ("soc_gkg", "SOC (g/kg)", "YlOrBr")]):
    for (county, gg), mk in zip(sites.groupby("county"), ["o", "^"]):
        sc = ax.scatter(gg.lon, gg.lat, c=gg[col], cmap=cmap, s=16, marker=mk,
                        edgecolor="k", linewidth=0.2, label=f"{county} (n={len(gg)})",
                        vmin=sites[col].min(), vmax=sites[col].max())
    fig.colorbar(sc, ax=ax, shrink=0.7, label=label)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{label}, 0-30 cm - {len(sites)} sites")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
    ax.grid(alpha=0.3)
fig.savefig("outputs/points_ph_soc_map.png", dpi=150)
print("\nWrote data/points_ph_soc_kakamega_transnzoia.csv and outputs/points_ph_soc_map.png")
