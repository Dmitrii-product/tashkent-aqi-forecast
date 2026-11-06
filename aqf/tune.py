"""Evolutionary search of LightGBM settings, separately for each horizon group.

(mu + lambda) evolution strategy: a population of settings, the best `MU` survive, children
come from uniform crossover of two parents plus Gaussian mutation in a normalised space.
The genome also switches holiday features and CAMS features (<= 24 h) on or off.

Fitness = MAE (AQI) on the validation winter 2024/25, with the model trained on data before
1 Oct 2024. The winners are then re-trained before 1 Oct 2025 / 1 Oct 2026 and scored on
winters the search never saw, against the current per-horizon models.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        .venv/bin/python -m aqf.tune
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm
import numpy as np
import pandas as pd

from . import aqi
from .calendar import offday_features
from .data import (PROC, RAW, load_prev_runs_ensemble, load_target, load_weather,
                   splice_forecast_weather)
from .features import Climatology, hourly_rows, pm_state, same_hour_mean, target_times
from .models import HOURLY_VARIANTS, _weights

TUNE_GROUPS = [(1, 1), (4, 6), (13, 18), (37, 48), (97, 120), (145, 168)]
NEIGHBOURS = {  # tuned group -> groups that take its settings
    (1, 1): [(1, 1), (2, 2), (3, 3)], (4, 6): [(4, 6), (7, 12)], (13, 18): [(13, 18), (19, 24)],
    (37, 48): [(25, 36), (37, 48), (49, 72)], (97, 120): [(73, 96), (97, 120)],
    (145, 168): [(121, 144), (145, 168)],
}
SPACE = {  # name: (kind, low, high) or (cat, options)
    "learning_rate": ("log", 0.01, 0.15),
    "num_leaves": ("logint", 7, 127),
    "min_child_samples": ("logint", 20, 3000),
    "n_estimators": ("logint", 100, 1200),
    "colsample_bytree": ("lin", 0.3, 1.0),
    "subsample": ("lin", 0.5, 1.0),
    "reg_lambda": ("log", 0.1, 100.0),
    "objective": ("cat", ["l2", "l1", "huber"]),
    "holidays": ("cat", [0, 1]),
    "cams": ("cat", [0, 1]),
}
MU, LAMBDA, GENERATIONS, INIT = 4, 8, 5, 9
OUT = PROC / "tuning.json"


# ------------------------------------------------------------------ genome <-> params
def decode(z: dict) -> dict:
    p = {}
    for k, spec in SPACE.items():
        v = z[k]
        if spec[0] == "cat":
            p[k] = spec[1][int(v) % len(spec[1])]
            continue
        _, lo, hi = spec
        v = min(max(v, 0.0), 1.0)
        if spec[0] == "lin":
            p[k] = lo + v * (hi - lo)
        else:
            x = np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
            p[k] = int(round(x)) if spec[0] == "logint" else float(x)
    return p


def random_genome(rng) -> dict:
    return {k: (rng.integers(len(s[1])) if s[0] == "cat" else rng.random()) for k, s in SPACE.items()}


def mutate(z: dict, rng, sigma: float = 0.15) -> dict:
    c = dict(z)
    for k, s in SPACE.items():
        if s[0] == "cat":
            if rng.random() < 0.2:
                c[k] = rng.integers(len(s[1]))
        else:
            c[k] = float(np.clip(c[k] + rng.normal(0, sigma), 0, 1))
    return c


def crossover(a: dict, b: dict, rng) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in SPACE}


def encode_current(group: tuple[int, int]) -> dict:
    """Genome of the setting now in production for this group (seeded into the search)."""
    choice = json.loads((PROC / "horizon_choice.json").read_text())
    v = choice[f"{group[0]}-{group[1]}"]
    extra = HOURLY_VARIANTS[v].get("lgbm_extra", {})
    rounds = HOURLY_VARIANTS[v].get("rounds", 300)
    p = {"learning_rate": extra.get("learning_rate", 0.05), "num_leaves": extra.get("num_leaves", 31),
         "min_child_samples": extra.get("min_child_samples", 500), "n_estimators": rounds,
         "colsample_bytree": 0.7, "subsample": 0.8, "reg_lambda": 10.0,
         "objective": extra.get("objective", "l2"), "holidays": 0,
         "cams": int(bool(HOURLY_VARIANTS[v].get("use_cams")))}
    z = {}
    for k, s in SPACE.items():
        if s[0] == "cat":
            z[k] = s[1].index(p[k])
        elif s[0] == "lin":
            z[k] = (p[k] - s[1]) / (s[2] - s[1])
        else:
            z[k] = (np.log(p[k]) - np.log(s[1])) / (np.log(s[2]) - np.log(s[1]))
    return z


# ------------------------------------------------------------------ design matrices
class Split:
    """Training rows (targets before `cut`) and validation rows (origins in [cut, end))."""

    def __init__(self, group, cut, end, wf_fc, cams, seed=0, val_step=3):
        lo, hi = group
        tgt, _ = load_target(profile_cutoff=cut)
        pm, src = tgt["pm"], tgt["src"]
        lp_all = np.log1p(pm)
        lp = lp_all[lp_all.index < cut]
        clim = Climatology().fit(lp)
        state, sh7 = pm_state(lp), same_hour_mean(lp)
        rng = np.random.default_rng(seed)
        origins = state.index[(state.index.hour % 2 == 0) & state["lag0"].notna()]
        k = min(4, hi - lo + 1)
        H = rng.integers(lo, hi + 1, size=(len(origins), k)) if hi > lo else np.full((len(origins), 1), lo)
        X = hourly_rows(origins, H, lp, state, sh7, clim, wf_fc, src, loo=True, cams=cams)
        T = target_times(X)
        y = lp.reindex(T).values - X["lag0"].values
        ok = ~np.isnan(y)
        self.X, self.y, T = X[ok], y[ok], T[ok]
        self.w = np.asarray(_weights(T, self.X["src"].values), float)
        self.X = pd.concat([self.X.reset_index(drop=True), offday_features(T)], axis=1)
        # validation: forecasts issued every `val_step` h inside the winter, all group horizons
        end = min(end, pm.dropna().index.max() - pd.Timedelta(hours=1))
        vo = pd.date_range(cut, end, freq=f"{val_step}h")
        vo = vo[lp_all.ffill(limit=2).reindex(vo).notna().values]
        st_all, sh_all = pm_state(lp_all), same_hour_mean(lp_all)
        Xv = hourly_rows(vo, np.arange(lo, hi + 1), lp_all, st_all, sh_all, clim, wf_fc, None, cams=cams)
        Xv["src"] = 1.0
        Tv = target_times(Xv)
        act = aqi.pm_to_aqi(pm.reindex(Tv).values)
        okv = ~np.isnan(act) & ~np.isnan(Xv["lag0"].values)
        self.Xv = pd.concat([Xv[okv].reset_index(drop=True), offday_features(Tv[okv])], axis=1)
        self.act = act[okv]
        self.cams_cols = [c for c in self.X.columns if c.startswith("cams_")]
        self.hol_cols = list(offday_features(Tv[:1]).columns)

    def columns(self, p: dict) -> list[str]:
        drop = set()
        if not p["cams"]:
            drop |= set(self.cams_cols)
        if not p["holidays"]:
            drop |= set(self.hol_cols)
        cols = [c for c in self.X.columns if c not in drop]
        return [c for c in cols if self.X[c].notna().any()]

    def score(self, p: dict, seed: int = 0) -> float:
        cols = self.columns(p)
        m = lightgbm.LGBMRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
            subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"], objective=p["objective"], n_jobs=8, random_state=seed,
            verbose=-1)
        m.fit(self.X[cols].values, self.y, sample_weight=self.w)
        z = m.predict(self.Xv[cols].values)
        pred = aqi.pm_to_aqi(np.expm1(z + self.Xv["lag0"].values))
        return float(np.nanmean(np.abs(pred - self.act)))


# ------------------------------------------------------------------ search
def evolve(split: Split, group, rng, log) -> list[tuple[float, dict]]:
    pop = [encode_current(group)] + [random_genome(rng) for _ in range(INIT - 1)]
    scored = []
    for i, z in enumerate(pop):
        f = split.score(decode(z))
        scored.append((f, z))
        log(f"    init {i}: {f:.2f}" + ("  (current production)" if i == 0 else ""))
    for g in range(GENERATIONS):
        scored.sort(key=lambda t: t[0])
        parents = scored[:MU]
        children = []
        for _ in range(LAMBDA):
            a, b = rng.choice(len(parents), 2, replace=False)
            children.append(mutate(crossover(parents[a][1], parents[b][1], rng), rng,
                                   sigma=0.2 * (0.8 ** g)))
        for z in children:
            scored.append((split.score(decode(z)), z))
        scored.sort(key=lambda t: t[0])
        log(f"    gen {g + 1}: best {scored[0][0]:.2f}  (current {split.score(decode(pop[0])):.2f})"
            if g == GENERATIONS - 1 else f"    gen {g + 1}: best {scored[0][0]:.2f}")
    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="")
    a = ap.parse_args()
    groups = [tuple(map(int, g.split("-"))) for g in a.groups.split(",")] if a.groups else TUNE_GROUPS
    wx = load_weather()
    wf_fc = splice_forecast_weather(wx, load_prev_runs_ensemble())
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")
    rng = np.random.default_rng(42)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    log = lambda s: print(s, flush=True)
    for grp in groups:
        t0 = time.time()
        log(f"group {grp[0]}-{grp[1]} h")
        val = Split(grp, pd.Timestamp("2024-10-01"), pd.Timestamp("2025-04-01"), wf_fc, cams)
        scored = evolve(val, grp, rng, log)
        best_p, cur_p = decode(scored[0][1]), decode(encode_current(grp))
        out = {"validation_best": scored[0][0], "validation_current": val.score(cur_p),
               "params": best_p, "current": cur_p, "tests": {}}
        for cut, end in (("2025-10-01", "2026-04-01"), ("2026-10-01", "2027-04-01")):
            te = Split(grp, pd.Timestamp(cut), pd.Timestamp(end), wf_fc, cams)
            out["tests"][cut[:4]] = {"tuned": te.score(best_p), "current": te.score(cur_p), "n": int(len(te.act))}
        res[f"{grp[0]}-{grp[1]}"] = out
        OUT.write_text(json.dumps(res, indent=1, default=str))
        log(f"  {grp}: validation {out['validation_current']:.2f} -> {out['validation_best']:.2f}; "
            f"test 2025/26 {out['tests']['2025']['current']:.2f} -> {out['tests']['2025']['tuned']:.2f}; "
            f"Oct 2026 {out['tests']['2026']['current']:.2f} -> {out['tests']['2026']['tuned']:.2f}; "
            f"{time.time() - t0:.0f}s; params {best_p}")


if __name__ == "__main__":
    main()
