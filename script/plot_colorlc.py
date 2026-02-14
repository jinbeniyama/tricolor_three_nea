#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create phased color curves for multiple objects.

Input file example: gri_col.txt (Seimei obs.)
  Columns include:
    "obj", "t_jd_ltcor", "g_r", "g_rerr", "r_i", "r_ierr", "alpha"
"""
import os
from argparse import ArgumentParser as ap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common import mycolor, mymark, plot_jbstyle, rotation_nea


def calc_color_significance(mean_list, err_list, obj=None, color=None, verbose=True):
    """
    Calculate color variation significance from binned means.

    Parameters
    ----------
    mean_list : array-like
        Mean color in each phase bin.
    err_list : array-like
        Error of mean color in each phase bin.
    obj : str, optional
        Object name.
    color : str, optional
        Color name (e.g., 'g-r').
    verbose : bool
        If True, print results.

    Returns
    -------
    dC : float
        Peak-to-peak color difference (Delta C_lc).
    err : float
        Uncertainty of the difference.
    """

    means = np.array(mean_list)
    errs  = np.array(err_list)

    if len(means) < 2:
        return np.nan, np.nan

    i_max = np.argmax(means)
    i_min = np.argmin(means)

    # Calculate observed color variation and its error
    dC = means[i_max] - means[i_min]
    err = np.sqrt(errs[i_max]**2 + errs[i_min]**2)

    if verbose:
        tag = ""
        if obj is not None:
            tag += f"{obj} "
        if color is not None:
            tag += f"{color} "

        print(f"\n  >>> {tag}color variation:")
        print(f"      ΔC = {dC:.3f} ± {err:.3f} (mag)")

    return dC, err


def chi2_constant_test(vals, errs, obj=None, color=None, verbose=True):
    """
    Chi-square test for constant color.

    Parameters
    ----------
    vals : array-like
        Color measurements.
    errs : array-like
        Measurement errors.
    obj : str
        Object name (optional).
    color : str
        Color name (optional).
    verbose : bool
        Print results if True.

    Returns
    -------
    chi2 : float
    dof : int
    redchi2 : float
    """

    import numpy as np
    from scipy.stats import chi2 as chi2_dist

    vals = np.array(vals)
    errs = np.array(errs)

    # weighted mean
    w = 1.0 / errs**2
    mean = np.sum(w * vals) / np.sum(w)

    # chi-square
    chi2 = np.sum((vals - mean)**2 / errs**2)
    dof = len(vals) - 1
    redchi2 = chi2 / dof

    # p-value
    pval = 1.0 - chi2_dist.cdf(chi2, dof)

    if verbose:
        tag = ""
        if obj:
            tag += f"{obj} "
        if color:
            tag += f"{color} "

        print(f"\n  >>> {tag}constant-color test:")
        print(f"      chi2 = {chi2:.2f}")
        print(f"      dof = {dof}")
        print(f"      reduced chi2 = {redchi2:.2f}")
        print(f"      p-value = {pval:.3e}")

        if pval < 0.003:
            print("      → significant color variation (~3σ or more)")
        elif pval < 0.05:
            print("      → marginal color variation")
        else:
            print("      → consistent with constant color")

    return chi2, dof, redchi2, pval


def calc_maximum_spot_size(C_main, C_spot, deltaC):
    """
    Calculate the maximum projected area ratio of a surface spot based on
    rotational color variation.

    Parameters
    ----------
    C_main : float
        Intrinsic color of the main (global) surface.
    C_spot : float
        Intrinsic color of the surface spot.
    deltaC : float
        The maximum color variation observed in the rotational lightcurve (Delta C_lc).

    Returns
    -------
    f_spot_max : float
        Maximum projected area ratio of the spot.
    """

    # Based on the two-component linear mixing model defined in the text:
    # C_obs(f_spot) = (1 - f_spot) * C_main + f_spot * C_spot

    # The observed difference in color lightcurve, Delta C_lc, is expressed as:
    # deltaC = | C_obs(f_spot_max) - C_obs(0) |
    #        = f_spot_max * | C_spot - C_main |

    # Thus, the maximum projected area ratio is:
    # f_spot_max = deltaC / | C_spot - C_main |

    # Example:
    # If C_main = 0.60, C_spot = 0.40, and deltaC = 0.05:
    # f_spot_max = 0.05 / |0.40 - 0.60| = 0.25 (25% of the projected surface)

    # Calculation
    color_contrast = np.abs(C_spot - C_main)

    # Avoid division by zero if assumed colors are identical
    if color_contrast == 0:
        return np.nan 

    f_spot_max = deltaC / color_contrast

    # Interpretation of f_spot_max > 1.0:
    # This indicates that the assumed color contrast |C_spot - C_main| is
    # insufficient to explain the observed deltaC, suggesting that the
    # spot material must have a more extreme color than assumed.
    if f_spot_max > 1.0:
        print(
            f"Calculated f_spot_max ({f_spot_max:.2f}) exceeds 1.0. "
            "This suggests the assumed color contrast |C_spot - C_main| is too small "
            "to account for the observed deltaC."
        )
        # f_spot_max = 1.0 # Optional: clip to 1.0 if necessary for further analysis


    return f_spot_max


def analyze_and_plot_color(
    ax, df, col_name, 
    label, col_color, marker, 
    c_a, c_b,
    value_shift=0.0, offset=0.0, y_text=0.0, 
    obj=None
):
    """
    Process data for a single color, calculate binned stats, and plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    df : pandas.DataFrame
        Data source.
    col_name : str
        Column name of color index.
    label : str
        Label for plotting.
    col_color : str
        Color code/name for plotting.
    marker : str
        Marker style.
    c_a : float
        Intrinsic color of component A (main).
    c_b : float
        Intrinsic color of component B (spot).
    value_shift : float
        Shift value to add to the color index (e.g., -mean).
    offset : float
        Additional offset for display convenience.
    y_text : float
        Y-position for text annotation.
    obj : str
        Object name for logging.
    """
    col_val = col_name
    col_err_name = f"{col_name}err" if f"{col_name}err" in df.columns else f"{col_name}_err"

    if col_val not in df.columns:
        return

    # Plot Scatter + error bars
    # Plot using df[col_val] + value_shift + offset
    ax.errorbar(
        df["phase"], df[col_val] + value_shift + offset,
        yerr=df[col_err_name],
        ms=5, color=col_color, marker=" ", capsize=0,
        ls="None", label=None, zorder=1, lw=0.5
    )
    ax.scatter(
        df["phase"], df[col_val] + value_shift + offset,
        marker=marker, s=50, color=col_color,
        facecolor="None", zorder=1, label=label, lw=0.5
    )

    # Binning
    nbin = 10
    width = 1. / nbin
    mean_list = []
    err_list  = []

    for n in range(nbin):
        p0, p1 = n * width, (n + 1) * width
        pmean = (p0 + p1) / 2.
        df_p = df[(df["phase"] >= p0) & (df["phase"] < p1)]

        if len(df_p) == 0:
            continue
        
        # Simple Mean and std err (for log)
        mean   = np.mean(df_p[col_val])
        std    = np.std(df_p[col_val])
        stderr = std / np.sqrt(len(df_p))
        print(f"  {col_name} {p0:.1f}–{p1:.1f}: mean={mean:.3f}, std={std:.3f}, stderr={stderr:.3f}")

        # Weighted mean and its err (for plot/calculation)
        vals = df_p[col_val].values
        errs = df_p[col_err_name].values
        weights = 1.0 / errs**2
        mean_w = np.sum(weights * vals) / np.sum(weights)
        stderr_w = np.sqrt(1.0 / np.sum(weights))
        print(f"  {col_name} {p0:.1f}–{p1:.1f}: mean={mean_w:.3f}, err={stderr_w:.3f}")

        mean_list.append(mean_w)
        err_list.append(stderr_w)

        # Plot weighted mean (shifted and offset)
        ax.errorbar(
            pmean, mean_w + value_shift + offset, stderr_w,
            color=col_color, marker=marker, ms=10,
            ls="None", label=None, zorder=2, lw=1.0, mec="black"
        )

        # Annotate with mean and stderr
        #ax.text(
        #    pmean, y_text,
        #    f"${mean_w + value_shift + offset:.3f}\pm{stderr_w:.3f}$",
        #    color="black", fontsize=14, ha="center", va="bottom", zorder=3
        #)

    # Significance calculation
    if len(mean_list) >= 2:
        dC, dCerr = calc_color_significance(mean_list, err_list, obj=obj, color=col_name)

        # 1-sigma
        dC_upper = dC + dCerr
        f_spot_max = calc_maximum_spot_size(c_a, c_b, dC_upper)
        print(f"  -> Maximum spot ratio: {f_spot_max:.2f}")




if __name__ == "__main__":
    parser = ap(
        description="Create phased color curves for multiple objects")
    parser.add_argument(
        "res", type=str, 
        help="Photometric results (gri_col.txt)")
    parser.add_argument(
        "res_GQ1", type=str, 
        help="Photometric results of GQ1")
    parser.add_argument(
        "--outdir", type=str, default="fig", 
        help="output directory")
    parser.add_argument(
        "--outtype", default="pdf", 
        help="format of output figure")
    args = parser.parse_args()

    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # LT corrected time key
    key_t = "t_jd_ltcor"

    # Colors and markers
    cols = ["g_r", "r_i"]
    colormap = {"g_r": mycolor[0], "r_i": mycolor[1]}
    markmap = {"g_r": mymark[0], "r_i": mymark[1]}

    # Read data
    df_all = pd.read_csv(args.res, sep=" ")
    JD0 = 2459503.1
    df_all[key_t] -= JD0

    plot_jbstyle()

    objtex = {
        "2021UW1": "2021 $UW_1$",
        "2021TY14": "2021 $TY_{14}$",
        "2022GQ1": "2022 $GQ_1$",
    }


    # For 2021UW1 and 2021TY14
    for obj in ["2021UW1", "2021TY14"]:
        obj_tex = objtex[obj]
        print(f"\nProcessing {obj} ...")
        df = df_all[df_all["obj"] == obj].copy()

        rotP_sec, _, _, _ = rotation_nea(obj)

        # Phase calculation
        df["phase"] = df[key_t] * 24. * 3600. / rotP_sec % 1

        # Setup figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.10, 0.16, 0.85, 0.75])

        magmin, magmax = 0.0, 1.5
        y0, y1 = magmin, magmax

        for idx, c in enumerate(cols):
            col = colormap[c]
            mark = markmap[c]

            if c == "g_r":
                label = "g-r"
                offset = 1.0 # Center at 1.0
                y_text = 1.25
                # Typical colors for S-type vs C-type or Q-type? 
                # Let's assume some defaults or pass specific ones if known
                c_main_assumed = 0.68 # typical S-type g-r
                c_spot_assumed = 0.43 # typical Q-type g-r
            else:
                label = "r-i"
                offset = 0.5 # Shift down by 0.5 relative to g-r (center at 0.5)
                y_text = 0.25
                c_main_assumed = 0.20 # typical S-type r-i
                c_spot_assumed = 0.10 # typical Q-type r-i

            col_val = c
            if col_val not in df.columns:
                continue

            # Calculate shift to center data properties around 0 before adding offset
            # For 2021UW1/2021TY14 which are absolute colors, 
            # if we want "relative variation centered at X", maybe we subtract mean and add offset?
            # The prompt says "center is 1".
            # If we want to see variation, usually we do (val - mean) + 1.0.
            gmean = np.mean(df[col_val])

            analyze_and_plot_color(
                ax, df, c, 
                label, col, mark, 
                c_a=c_main_assumed, c_b=c_spot_assumed,
                value_shift=-gmean, offset=offset, y_text=y_text, 
                obj=obj
            )

        ax.set_xlabel(f"Rotation phase (P={rotP_sec} s, JD0={JD0})")
        ax.set_ylabel("Color index")
        #ax.legend(ncol=2, loc="upper left")
        ax.set_xlim([0, 1])
        ax.set_ylim([y0, y1])
        ax.set_title(obj_tex)

        # Save per target
        out = os.path.join(outdir, f"{obj}_colorlc.{args.outtype}")
        fig.savefig(out)
        plt.close()
        print(f"  -> Saved: {out}")



    # For 2022 GQ1
    for obj in sorted(["2022GQ1"]):
        print(f"\nProcessing {obj} ...")
        obj_tex = objtex[obj]

        df = pd.read_csv(args.res_GQ1, sep=" ")

        rad_GQ1 = 10
        rad_GQ1 = 10
        df = df[df["radius"] == rad_GQ1]

        rotP_sec, _, _, _ = rotation_nea(obj)

        # Phase calculation
        df["phase"] = df[key_t] * 24. * 3600. / rotP_sec % 1

        # Setup figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.10, 0.16, 0.85, 0.75])
 
        # Use mag. This is not ideal when S/N is low. ========================= 
        # The noise distribution is not  gaussian.
        magmin, magmax = 0.0, 1.5
        y0, y1 = magmin, magmax

        for idx, c in enumerate(cols):
            col = colormap[c]
            mark = markmap[c]

            if c == "g_r":
                label = "g-r"
                offset = 1.0
                y_text = 1.25
                c_main_assumed = 0.40 # Q-type g-r? just example
                c_spot_assumed = 0.60
            else:
                label = "r-i"
                offset = 0.5
                y_text = 0.25
                c_main_assumed = 0.05
                c_spot_assumed = 0.15


            col_val = c
            if col_val not in df.columns:
                continue
            
            # Global mean
            gmean = np.mean(df[col_val])

            analyze_and_plot_color(
                ax, df, c, 
                label, col, mark, 
                c_a=c_main_assumed, c_b=c_spot_assumed,
                value_shift=-gmean, offset=offset, y_text=y_text, 
                obj=obj
            )

        ax.set_xlabel(f"Rotation phase (P={rotP_sec} s, JD0={JD0})")
        ax.set_ylabel("Color index")
        #ax.legend(ncol=2, loc="upper left")
        ax.set_xlim([0, 1])
        ax.set_ylim([y0, y1])
        ax.set_title(obj_tex)
        # Use mag. This is not ideal when S/N is low. ========================= 



        # Use flux ============================================================
        # For 2022 GQ1 (flux ratio-based)
        #df["g_r_ratio"] = df["flux_g"] / df["flux_r"]
        #df["g_r_ratio_err"] = df["g_r_ratio"] * np.sqrt(
        #    (df["fluxerr_g"]/df["flux_g"])**2 + (df["fluxerr_r"]/df["flux_r"])**2
        #)
        #df["r_i_ratio"] = df["flux_r"] / df["flux_i"]
        #df["r_i_ratio_err"] = df["r_i_ratio"] * np.sqrt(
        #    (df["fluxerr_r"]/df["flux_r"])**2 + (df["fluxerr_i"]/df["flux_i"])**2
        #)
        #
        ## --- calculation: shift center to 1.0 ---
        #df["g_r_calc"] = df["g_r_ratio"] / np.mean(df["g_r_ratio"])
        #df["g_r_calc_err"] = df["g_r_ratio_err"] / np.mean(df["g_r_ratio"])
        #df["r_i_calc"] = df["r_i_ratio"] / np.mean(df["r_i_ratio"])
        #df["r_i_calc_err"] = df["r_i_ratio_err"] / np.mean(df["r_i_ratio"])
        #
        ## --- plot setup ---
        #fig, ax = plt.subplots(figsize=(16,5))
        #ax.set_xlim([0,1])
        #ax.set_ylim([0.3, 3.5])
        #ax.set_xlabel(f"Rotation phase (P={rotP_sec} s, JD0={JD0})")
        #ax.set_ylabel("Flux ratio (shifted)")
        #
        ## Plot offset: only for display
        #plot_offsets = {"g_r_calc": 1.0, "r_i_calc": 0.0}  # will add 1/2 for final positions
        #display_centers = {"g_r_calc": 2.0, "r_i_calc": 1.0}
        #
        #ratios = [
        #    ("g_r_calc", "g_r_calc_err", "g_r_ratio", "g-r"),
        #    ("r_i_calc", "r_i_calc_err", "r_i_ratio", "r-i")
        #]
        #
        #for idx, (col_calc, col_calc_err, col_plot_orig, label) in enumerate(ratios):

        #    # offset for plotting only
        #    offset = display_centers[col_calc] - 1.0
        #    

        #
        #    # Scatter + error bars
        #    ax.errorbar(
        #        df["phase"], df[col_calc] + offset,
        #        yerr=df[col_calc_err],
        #        ms=5, color=colormap.get(label,f"C{idx}"), marker=" ", capsize=0,
        #        ls="None", label=None, zorder=1, lw=0.7
        #    )
        #    ax.scatter(
        #        df["phase"], df[col_calc] + offset,
        #        marker=markmap.get(label,"o"), s=50,
        #        color=colormap.get(label,f"C{idx}"),
        #        facecolor="None", zorder=1, label=label, lw=0.7
        #    )
        #
        #    # Phase binning + weighted mean
        #    nbin = 10
        #    width = 1./nbin
        #    mean_list, err_list = [], []
        #
        #    for n in range(nbin):
        #        p0, p1 = n*width, (n+1)*width
        #        pmean = (p0+p1)/2.
        #        df_p = df[(df["phase"] >= p0) & (df["phase"] < p1)]
        #        if len(df_p)==0:
        #            continue
        #
        #        vals = df_p[col_calc].values
        #        errs = df_p[col_calc_err].values
        #        weights = 1.0 / errs**2
        #        mean = np.sum(weights*vals)/np.sum(weights)
        #        stderr = np.sqrt(1.0/np.sum(weights))
        #
        #        mean_list.append(mean)
        #        err_list.append(stderr)
        #
        #        # plot weighted mean
        #        ax.errorbar(
        #            pmean, mean + offset, stderr,
        #            color="orange", marker="o", ms=10,
        #            ls="None", label=None, zorder=2, lw=0.7, mec="black"
        #        )
        #        # annotate with calculation value (center 1.0)
        #        ax.text(
        #            pmean, display_centers[col_calc]+0.2,  # y位置は見やすく offset
        #            f"${mean:.2f}\pm{stderr:.2f}$",        # text は offset なし
        #            fontsize=12, ha="center", va="bottom"
        #        )
        #
        #    # --- significance / χ² ---
        #    if len(mean_list)>=2:
        #        calc_color_significance(mean_list, err_list, obj=obj, color=label)
        #        chi2_constant_test(df[col_calc].values, df[col_calc_err].values, obj=obj, color=label)
        # Use flux ============================================================


        # Save per target
        out = os.path.join(outdir, f"{obj}_colorlc.{args.outtype}")
        fig.savefig(out)
        plt.close()
        print(f"  -> Saved: {out}")
