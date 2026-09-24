"""
Step 3 - Quantile Regression Forest for pH and SOC (Kakamega + Trans Nzoia)

Inputs (relative to the project folder)
  data/points_ph_soc_kakamega_transnzoia.csv        from step 1
  data/covariates/cov_kak_tnz_100m*.tif             from step 2 (tiles are merged)
  data/covariates/cov_kak_tnz_100m_bands.csv        band order from step 2

Outputs
  outputs/model/cv_summary.csv            spatial vs random CV metrics
  outputs/model/<prop>_cv_predictions.csv
  outputs/model/<prop>_importance.csv
  outputs/model/<prop>_cv_plot.png        observed vs predicted
  outputs/maps/<prop>_0_30cm.tif          6 bands: q05, q50, q95, pi90_width, di, aoa
  outputs/maps/<prop>_0_30cm.png          quick-look map

Run from the repo root:  python scripts/03_model_predict.py   (or %run in Colab/Jupyter)
Needs: pip install quantile-forest rasterio pyproj scikit-learn scipy pandas matplotlib
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from quantile_forest import RandomForestQuantileRegressor
from rasterio.merge import merge
from rasterio.windows import Window
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.model_selection import GroupKFold, KFold

# ------------------------------------------------------------------ settings
POINTS = "data/points_ph_soc_kakamega_transnzoia.csv"
COV_PATTERN = "data/covariates/cov_kak_tnz_100m*.tif"
COV_MERGED = "data/covariates/cov_kak_tnz_100m_merged.tif"
BANDS_CSV = "data/covariates/cov_kak_tnz_100m_bands.csv"
PROPS = {"ph": "pH", "soc_gkg": "SOC (g/kg)"}
NODATA = -9999

BLOCK_M = 10_000       # spatial CV block size; larger than the Trans Nzoia clusters
N_FOLDS = 5
QUANTILES = [0.05, 0.5, 0.95]
RF = dict(n_estimators=500, min_samples_leaf=3, max_features=0.33,
          random_state=42, n_jobs=-1)
ROW_CHUNK = 128        # raster rows predicted at a time (lower if memory is tight)

OUT_M, OUT_MAPS = Path("outputs/model"), Path("outputs/maps")
OUT_M.mkdir(parents=True, exist_ok=True)
OUT_MAPS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ covariates
def covariate_raster():
    """Return the covariate GeoTIFF path, merging GEE export tiles if needed."""
    tiles = sorted(p for p in glob.glob(COV_PATTERN) if "merged" not in p)
    if not tiles:
        raise FileNotFoundError(f"No covariate GeoTIFF matching {COV_PATTERN}")
    if len(tiles) == 1:
        return tiles[0]
    if not Path(COV_MERGED).exists():
        print(f"Merging {len(tiles)} export tiles ...")
        srcs = [rasterio.open(t) for t in tiles]
        mosaic, transform = merge(srcs, nodata=NODATA)
        prof = srcs[0].profile
        prof.update(height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=transform, nodata=NODATA, compress="deflate",
                    tiled=True, BIGTIFF="IF_SAFER")
        with rasterio.open(COV_MERGED, "w", **prof) as dst:
            dst.write(mosaic)
        for s in srcs:
            s.close()
    return COV_MERGED


COV = covariate_raster()
band_names = (pd.read_csv(BANDS_CSV).sort_values("band_index")["band_name"].tolist())

sites = pd.read_csv(POINTS)
with rasterio.open(COV) as src:
    assert src.count == len(band_names), (
        f"Raster has {src.count} bands but band list has {len(band_names)}")
    to_rast = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    xs, ys = to_rast.transform(sites.lon.values, sites.lat.values)
    X_all = np.array(list(src.sample(zip(xs, ys))), dtype="float64")
    profile = src.profile
    raster_crs = src.crs

valid = ~(X_all == NODATA).any(axis=1) & np.isfinite(X_all).all(axis=1)
print(f"Sites: {len(sites)}  with valid covariates: {valid.sum()}")
sites = sites[valid].reset_index(drop=True)
X_all = X_all[valid]
sites["x"], sites["y"] = xs[valid], ys[valid]

# spatial blocks: sites in the same block are never split between train and test
blk = (np.floor(sites.x / BLOCK_M).astype(int).astype(str) + "_" +
       np.floor(sites.y / BLOCK_M).astype(int).astype(str))
sites["block"] = pd.factorize(blk)[0]
print(f"Spatial blocks ({BLOCK_M/1000:.0f} km): {sites.block.nunique()}")


# ------------------------------------------------------------------ helpers
def qrf():
    return RandomForestQuantileRegressor(**RF)


def metrics(obs, q):
    lo, med, hi = q[:, 0], q[:, 1], q[:, 2]
    err = med - obs
    r = np.corrcoef(obs, med)[0, 1]
    ccc = (2 * r * obs.std() * med.std() /
           (obs.var() + med.var() + (obs.mean() - med.mean()) ** 2))
    return {"n": len(obs), "ME": err.mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "MAE": np.abs(err).mean(),
            "R2_MEC": 1 - (err ** 2).sum() / ((obs - obs.mean()) ** 2).sum(),
            "CCC": ccc, "PICP90": ((obs >= lo) & (obs <= hi)).mean()}


def cross_validate(X, y, splitter, groups=None):
    oof = np.full((len(y), len(QUANTILES)), np.nan)
    for tr, te in splitter.split(X, y, groups):
        oof[te] = qrf().fit(X[tr], y[tr]).predict(X[te], quantiles=QUANTILES)
    return oof


class AOA:
    """Area of Applicability (Meyer & Pebesma 2021): dissimilarity index (DI)
    in importance-weighted, standardised covariate space."""

    def __init__(self, X, importance, folds):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        self.w = importance / importance.sum()
        Z = self._scale(X)
        self.mean_dist = pdist(Z).mean()
        self.tree = cKDTree(Z)
        # training DI: distance to nearest training point in *other* CV folds
        di_train = np.empty(len(Z))
        for f in np.unique(folds):
            inf, outf = folds == f, folds != f
            d, _ = cKDTree(Z[outf]).query(Z[inf])
            di_train[inf] = d / self.mean_dist
        q1, q3 = np.percentile(di_train, [25, 75])
        self.threshold = min(q3 + 1.5 * (q3 - q1), di_train.max())

    def _scale(self, X):
        return (X - self.mu) / self.sd * self.w

    def di(self, X):
        d, _ = self.tree.query(self._scale(X))
        return d / self.mean_dist


# ------------------------------------------------------------------ modelling
summary = []
for prop, label in PROPS.items():
    d = sites.dropna(subset=[prop])
    X, y = X_all[d.index], d[prop].values
    print(f"\n=== {label}: n = {len(y)}")

    # spatial CV (honest) vs random CV (optimistic) - the gap shows clustering bias
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_sp = cross_validate(X, y, gkf, d.block.values)
    oof_rn = cross_validate(X, y, KFold(N_FOLDS, shuffle=True, random_state=42))
    for name, oof in [("spatial", oof_sp), ("random", oof_rn)]:
        m = {"property": prop, "cv": name, **metrics(y, oof)}
        summary.append(m)
        print(f"  {name:8s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()
                                           if isinstance(v, float)))

    out = d[["site_id", "county", "lon", "lat", prop]].copy()
    out[["q05", "q50", "q95"]] = oof_sp
    out.to_csv(OUT_M / f"{prop}_cv_predictions.csv", index=False)

    # observed vs predicted plot (spatial CV)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for county, g in out.groupby("county"):
        ax.scatter(g[prop], g.q50, s=12, alpha=0.7, label=county)
    lims = [min(y.min(), oof_sp[:, 1].min()), max(y.max(), oof_sp[:, 1].max())]
    ax.plot(lims, lims, "k--", lw=1)
    ms = [s for s in summary if s["property"] == prop and s["cv"] == "spatial"][0]
    ax.set_title(f"{label} - spatial CV\nR² {ms['R2_MEC']:.2f}  RMSE {ms['RMSE']:.2f}  "
                 f"PICP90 {ms['PICP90']:.2f}")
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted (median)"); ax.legend()
    fig.savefig(OUT_M / f"{prop}_cv_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # final model on all data
    model = qrf().fit(X, y)
    imp = pd.Series(model.feature_importances_, index=band_names).sort_values(ascending=False)
    imp.to_csv(OUT_M / f"{prop}_importance.csv", header=["importance"])
    print("  top covariates:", ", ".join(imp.index[:5]))

    folds = np.empty(len(y), dtype=int)
    for i, (_, te) in enumerate(gkf.split(X, y, d.block.values)):
        folds[te] = i
    aoa = AOA(X, model.feature_importances_, folds)

    # ---- predict the raster
    out_path = OUT_MAPS / f"{prop}_0_30cm.tif"
    prof = profile.copy()
    prof.update(count=6, dtype="float32", nodata=NODATA, compress="deflate",
                tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(COV) as src, rasterio.open(out_path, "w", **prof) as dst:
        for row in range(0, src.height, ROW_CHUNK):
            win = Window(0, row, src.width, min(ROW_CHUNK, src.height - row))
            arr = src.read(window=win).reshape(src.count, -1).T.astype("float64")
            ok = ~(arr == NODATA).any(axis=1) & np.isfinite(arr).all(axis=1)
            res = np.full((arr.shape[0], 6), NODATA, dtype="float32")
            if ok.any():
                p = model.predict(arr[ok], quantiles=QUANTILES)
                di = aoa.di(arr[ok])
                res[ok, :3] = p
                res[ok, 3] = p[:, 2] - p[:, 0]
                res[ok, 4] = di
                res[ok, 5] = (di <= aoa.threshold).astype("float32")
            dst.write(res.T.reshape(6, win.height, win.width), window=win)
            print(f"\r  predicting rows {row + win.height}/{src.height}", end="")
        dst.descriptions = ("q05", "q50", "q95", "pi90_width", "di", "aoa")
    print(f"\n  -> {out_path}")

    # ---- quick-look map
    with rasterio.open(out_path) as r:
        med = np.ma.masked_equal(r.read(2), NODATA)
        width = np.ma.masked_equal(r.read(4), NODATA)
        aoa_b = np.ma.masked_equal(r.read(6), NODATA)
        ext = [r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    panels = [(med, f"{label} - median prediction", "RdYlBu" if prop == "ph" else "YlOrBr"),
              (width, "90% prediction interval width", "magma_r"),
              (aoa_b, f"Area of applicability - white = inside (DI ≤ {aoa.threshold:.2f})", "Greys_r")]
    for ax, (img, title, cmap) in zip(axes, panels):
        im = ax.imshow(img, extent=ext, cmap=cmap)
        ax.scatter(d.x, d.y, s=2, c="k")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        if title.startswith("Area"):
            continue
        fig.colorbar(im, ax=ax, shrink=0.8)
    inside = float(aoa_b.mean())
    fig.suptitle(f"{label}, 0-30 cm   |   {inside:.0%} of the area inside the AOA")
    fig.savefig(OUT_MAPS / f"{prop}_0_30cm.png", dpi=130)
    plt.close(fig)

pd.DataFrame(summary).round(3).to_csv(OUT_M / "cv_summary.csv", index=False)
print("\nCV summary\n", pd.DataFrame(summary).round(3).to_string(index=False))
