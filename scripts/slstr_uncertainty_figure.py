"""Error-consistency (z-score) validation figures.

For a retrieved quantity x with reported uncertainties on both sides,
   delta = (x_orac - x_ref) / sqrt(sigma_orac^2 + sigma_ref^2)
should be ~ N(0,1) if the uncertainties are calibrated and errors Gaussian.

Each figure:
  (a) violin plots of delta binned by x_ref, with the +/-1 band and per-bin
      robust std annotated;
  (b) per-bin std (raw + robust IQR/1.349) vs x_ref, reference line at 1;
  (c) QQ plot of delta vs the standard normal, with skew / kurtosis / coverage.

Produces the water-COT figure (two-sided sigma) and the CTH figure (ATLID as
truth; ORAC sigma only, informative subset).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CACHE = Path(".uncertainty_cache")
OUT = Path("figures/slstr_uncertainty_2025-12")
NAVY = "#1b4f72"; RED = "#c0392b"; BLUE = "#2e6da4"; GREEN = "#1b7837"


def _robust_std(a):
    return (np.percentile(a, 75) - np.percentile(a, 25)) / 1.349


def _qq(ax, dl, color, label, lim=4):
    dl = dl[np.isfinite(dl)]
    dl = dl[np.abs(dl) < np.percentile(np.abs(dl), 99.5)]  # trim extreme for display
    qs = np.linspace(0.005, 0.995, 200)
    ax.plot(stats.norm.ppf(qs), np.quantile(dl, qs), "-", color=color, lw=1.8, label=label)


def error_consistency_fig(delta, xref, edges, *, xlabel, title, fname,
                          xscale="log", clip=6.0, split=None,
                          split_labels=("low", "high")):
    d = pd.DataFrame({"delta": delta, "xref": xref}).replace(
        [np.inf, -np.inf], np.nan).dropna()
    d["b"] = pd.cut(d["xref"], edges)
    groups, centers, rstd, std, ns = [], [], [], [], []
    for b, g in d.groupby("b", observed=True):
        if len(g) < 30:
            continue
        groups.append(np.clip(g["delta"].values, -clip, clip))
        centers.append(0.5 * (b.left + b.right))
        rstd.append(_robust_std(g["delta"].values))
        std.append(g["delta"].std())
        ns.append(len(g))
    centers = np.array(centers)
    # colour bins by low/high when a split is given
    bincol = [GREEN if (split is not None and c < split) else
              (RED if split is not None else BLUE) for c in centers]

    fig = plt.figure(figsize=(14, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])

    # (a) violins by x_ref bin, coloured low/high
    pos = np.arange(len(groups))
    ax0.axhspan(-1, 1, color=NAVY, alpha=0.10, zorder=0, label="±1 (ideal 68%)")
    ax0.axhline(0, color="k", lw=0.8)
    vp = ax0.violinplot(groups, positions=pos, showextrema=False, widths=0.85)
    for b, c in zip(vp["bodies"], bincol):
        b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor("0.3")
    ax0.plot(pos, [np.median(g) for g in groups], "o", color=NAVY, ms=4, zorder=5)
    for i, rs in enumerate(rstd):
        ax0.text(i, clip*0.92, f"{rs:.1f}", ha="center", fontsize=7.5,
                 color=bincol[i])
    ax0.text(-0.4, clip*0.92, "robust\nstd→", ha="right", va="center", fontsize=7, color="0.3")
    if split is not None:
        xb = np.searchsorted(centers, split) - 0.5
        ax0.axvline(xb, color="0.4", ls="--", lw=1)
        ax0.text(xb/2, -clip*0.7, split_labels[0].upper(), color=GREEN,
                 ha="center", fontsize=9, fontweight="bold")
        ax0.text((xb+len(groups)-0.5)/2, -clip*0.7, split_labels[1].upper(),
                 color=RED, ha="center", fontsize=9, fontweight="bold")
    ax0.set_xticks(pos)
    ax0.set_xticklabels([f"{c:.1f}" for c in centers], rotation=45, fontsize=8)
    ax0.set_ylim(-clip, clip); ax0.set_xlabel(xlabel); ax0.set_ylabel("normalised discrepancy δ")
    ax0.set_title("(a) δ distribution by reference value")
    ax0.legend(loc="lower right", fontsize=8)

    # (b) per-bin std vs x_ref, with low/high shading
    if split is not None:
        ax1.axvspan(min(centers)*0.5, split, color=GREEN, alpha=0.07)
        ax1.axvspan(split, max(centers)*1.5, color=RED, alpha=0.07)
        ax1.axvline(split, color="0.4", ls="--", lw=1)
    ax1.axhline(1, color="k", ls="--", lw=1, label="ideal (std=1)")
    ax1.plot(centers, std, "s-", color="0.45", ms=5, label="std")
    ax1.plot(centers, rstd, "o-", color=NAVY, ms=5, label="robust std")
    if xscale == "log":
        ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("std of δ per bin")
    ax1.set_title("(b) Is std(δ) ≈ 1?"); ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    # (c) QQ — split low vs high when requested, else single
    lim = 4
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="N(0,1) ideal")
    if split is not None:
        lo = d.loc[d["xref"] < split, "delta"].values
        hi = d.loc[d["xref"] >= split, "delta"].values
        _qq(ax2, lo, GREEN, f"{split_labels[0]} cloud")
        _qq(ax2, hi, RED, f"{split_labels[1]} cloud")
        txt = (f"{split_labels[0]}:  std {_robust_std(lo):.1f}  "
               f"±1 {100*(np.abs(lo)<1).mean():.0f}%\n"
               f"{split_labels[1]}: std {_robust_std(hi):.1f}  "
               f"±1 {100*(np.abs(hi)<1).mean():.0f}%")
        ax2.set_title("(c) Low vs high cloud (QQ)")
    else:
        _qq(ax2, d["delta"].values, NAVY, "δ")
        full = d["delta"].values
        txt = (f"N={len(full):,}\nmedian {np.median(full):+.2f}\n"
               f"robust std {_robust_std(full):.2f}\n"
               f"skew {stats.skew(full):+.1f}  ex-kurt {stats.kurtosis(full):+.1f}\n"
               f"within ±1: {100*(np.abs(full)<1).mean():.0f}%")
        ax2.set_title("(c) Normality (QQ)")
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("standard-normal quantile"); ax2.set_ylabel("δ quantile")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.text(0.03, 0.97, txt, transform=ax2.transAxes, va="top", fontsize=8,
             family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / fname
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)
    # printed summary
    print(f"  overall: median {np.median(d['delta']):+.2f}  robust std {_robust_std(d['delta'].values):.2f}"
          f"  within±1 {100*(np.abs(d['delta'])<1).mean():.0f}%  skew {stats.skew(d['delta']):+.1f}")


def do_cot():
    d = pd.read_parquet(CACHE / "cot_unc_pairs.parquet")
    m = ((d.quality_status_atlid == 0) & (d.liquid_only_atlid == True)
         & (d.cldmask_orac == 1) & (d.phase_orac == 1)
         & np.isfinite(d.cot_orac) & np.isfinite(d.cot_water_atlid)
         & np.isfinite(d.cot_unc_orac) & np.isfinite(d.cot_fracerr_atlid)
         & (d.cot_orac > 0) & (d.cot_water_atlid > 0) & (d.cot_unc_orac > 0))
    v = d[m].copy()
    sref = v.cot_fracerr_atlid * v.cot_water_atlid
    denom = np.sqrt(v.cot_unc_orac**2 + sref**2)
    delta = (v.cot_orac - v.cot_water_atlid) / denom
    edges = np.array([0.3, 1, 2, 3, 5, 7, 10, 15, 22, 32, 60])
    print("\n===== WATER-COT error consistency =====")
    error_consistency_fig(delta.values, v.cot_water_atlid.values, edges,
        xlabel="ACM-CAP liquid optical depth τ (reference)",
        title="ORAC SLSTR water-COT uncertainty consistency vs EarthCARE ACM-CAP "
              "— Antarctic daytime, Dec-2025\nδ = (τ_ORAC − τ_ref) / √(σ_ORAC² + σ_ref²)",
        fname="cot_error_consistency.png", xscale="log")


def do_cth():
    d = pd.read_parquet(CACHE / "cth_unc.parquet")
    cap = np.nanmax(d.cth_orac_corrected_uncertainty_km)
    sg = d.cth_orac_corrected_uncertainty_km
    m = (np.isfinite(d.cth_orac_corrected_km) & np.isfinite(d.cth_atlid_thick_km)
         & np.isfinite(sg) & (sg > 0.05) & (sg < 0.975*cap))
    v = d[m].copy()
    delta = (v.cth_orac_corrected_km - v.cth_atlid_thick_km) / v.cth_orac_corrected_uncertainty_km
    edges = np.array([0, 2, 4, 6, 8, 10, 12, 16])
    print("\n===== CTH error consistency (informative-σ subset; ATLID=truth) =====")
    error_consistency_fig(delta.values, v.cth_atlid_thick_km.values, edges,
        xlabel="ATLID cloud-top height (km, reference)",
        title="ORAC SLSTR CTH uncertainty consistency vs EarthCARE A-CTH "
              "— informative-σ subset (σ not at 20 km a-priori cap)\n"
              "δ = (CTH_ORAC − CTH_ATLID) / σ_ORAC · split at 6 km",
        fname="cth_error_consistency.png", xscale="linear", clip=6.0,
        split=6.0, split_labels=("low <6 km", "high ≥6 km"))
    # explicit low/high summary for the report
    xr = v.cth_atlid_thick_km.values
    for nm, msk in [("low  (<6 km)", xr < 6), ("high (≥6 km)", xr >= 6)]:
        g = delta.values[msk]; g = g[np.isfinite(g)]
        print(f"  {nm}: N={len(g):>6}  median {np.median(g):+.2f}  robust std {_robust_std(g):.1f}"
              f"  within±1 {100*(np.abs(g)<1).mean():.0f}%")


def main():
    if (CACHE / "cot_unc_pairs.parquet").exists():
        do_cot()
    if (CACHE / "cth_unc.parquet").exists():
        do_cth()


if __name__ == "__main__":
    raise SystemExit(main())
