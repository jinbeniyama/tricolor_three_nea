#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Useful info. and functions for tricolor photometry paper.
"""
import os
import pandas as pd
import numpy as np
import datetime
from astroquery.jplhorizons import Horizons
import matplotlib.pyplot as plt  
from astropy import units as u
from astropy.time import Time
from matplotlib.dates import DateFormatter, HourLocator, DayLocator, MonthLocator
from scipy.signal import argrelmax
from astropy.timeseries import LombScargle
from scipy.stats import gaussian_kde


# Constants and our results ===================================================
loc_Seimei = {
    "lon":        133.5967, 
    "lat":         34.5769, 
    "elevation" :   0.355
    }

# TODO: Check before submission 
def rotation_nea(obj):
    """Return rotation parameters.

    Parameter
    ---------
    obj : str
        object name

    Returns
    -------
    P, Perr, dm, dmerr : float
        rotation period, lightcurve amplitude, and their uncertainties
    """
    # All Pan-STARRS
    if obj == "2021TY14":
        P, Perr   = 15.282, 0.001
        dm, dmerr = 0.768, 0.012
    elif obj == "2021UW1":
        P, Perr   = 21.099, 0.003
        dm, dmerr = 0.242, 0.013
    elif obj == "2022GQ1":
        P, Perr   = 8.778, 0.012
        dm, dmerr = 0.768, 0.012
    return P, Perr, dm, dmerr

# TODO: Check before submission 
def color_nea(obj):
    """Return visible colors.

    Parameter
    ---------
    obj : str
        object name

    Returns
    -------
    gr, grerr, ri, rierr : float
        g-r and r-i colors and their uncertainties
    """
    # All Pan-STARRS
    if obj == "2021TY14":
        gr, grerr = 0.448, 0.003
        ri, rierr = 0.189, 0.002
    elif obj == "2021UW1":
        gr, grerr = 0.571, 0.005
        ri, rierr = 0.240, 0.002
    # This is calibrated by hand, see photometry_2022GQ1.ipynb
    elif obj == "2022GQ1":
        gr, grerr = 0.562, 0.041
        ri, rierr = 0.179, 0.033
    return gr, grerr, ri, rierr
# Constants and our results ===================================================


# To make table ===============================================================
def add_obsinfo(df, loc):
    """Add obs. info.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 
         'obj', 'obsdate0', 'obsdate1', 'code', 
    loc : str or dict
        location for JPL query

    Return
    ------
    df : pandas.DataFrame
        'H', 'V0', 'V1', 'elev0', 'elev1', 'vnorm0', 'vnorm1', 'r0', 'r1',
        'delta0', 'delta1', 'alpha0', 'alpha1' 
        added 15-columns DataFrame
    """

    columns = df.columns.tolist()
    assert "obj" in columns, "Check input."
    assert "obsdate0" in columns, "Check input."
    assert "obsdate1" in columns, "Check input."

    
    df = df.assign(
        H=0.0, V0=0.0, V1=0.0, elev0=0.0, elev1=0.0, vnorm0=0.0, vnorm1=0.0,
        r0=0.0, r1=0.0, delta0=0.0, delta1=0.0, alpha0=0.0, alpha1=0.0)
    for idx, row in df.iterrows():
        print(f"  Query {idx+1:02d} start")
        obj = row["obj"]
        try:
            t_start = datetime.datetime.strptime(row["obsdate0"], "%Y-%m-%dT%H:%M:%S.%f")
            t_end = datetime.datetime.strptime(row["obsdate1"], "%Y-%m-%dT%H:%M:%S.%f")
        except:
            t_start = datetime.datetime.strptime(row["obsdate0"], "%Y-%m-%dT%H:%M:%S")
            t_end = datetime.datetime.strptime(row["obsdate1"], "%Y-%m-%dT%H:%M:%S")
        t_isot = datetime.datetime.strftime(t_start, "%Y-%m-%dT%H:%M:%S")
        t_start = datetime.datetime.strftime(t_start, "%Y-%m-%d %H:%M:%S")
        t_end = datetime.datetime.strftime(t_end, "%Y-%m-%d %H:%M:%S")
        obj = f"{obj}"

        # Query with a single time
        if t_start == t_end:
            # Convert to JD
            t = Time(str(t_isot), format='isot', scale='utc')
            jpl = Horizons(id=obj, location=loc, epochs=t.jd)
        # Obtain an ephemeris with 1 min step
        else:
            jpl = Horizons(id=obj, location=loc,
            epochs={'start':t_start, 'stop':t_end, 'step':"1m"})
        eph= jpl.ephemerides()
        df.at[idx, "H"] = eph["H"][0]
        # Visible magnitude
        df.at[idx, "V0"] = eph["V"][0]
        df.at[idx, "V1"] = eph["V"][-1]
        # Elevation
        df.at[idx, "elev0"] = eph["EL"][0]
        df.at[idx, "elev1"] = eph["EL"][-1]
        # minimum/maximum airmass (do not corresponds to first and last frames)
        airmass = 1/np.cos(np.radians(90-eph["EL"]))
        df.at[idx, "airmass_min"] = np.min(airmass)
        df.at[idx, "airmass_max"] = np.max(airmass)
        dcos = np.cos(np.radians(eph["DEC"][0]))

        # Already arcsec/hour in sky motion. Do not need cos correction
        vnorm0 = (eph["RA_rate"][0]**2 + eph["DEC_rate"][0]**2)**0.5
        vnorm1 = (eph["RA_rate"][-1]**2 + eph["DEC_rate"][-1]**2)**0.5
        # arcsec/hour to arcsec/s
        df.at[idx, "vnorm0"] = vnorm0/3600.
        df.at[idx, "vnorm1"] = vnorm1/3600.

        #  'PsAng   PsAMV' =
        #      The position angles of the extended Sun-to-target radius vector ("PsAng")
        #  and the negative of the targets' heliocentric velocity vector ("PsAMV"), as
        #  seen in the observers' plane-of-sky, measured counter-clockwise (east) from
        #  reference-frame north-pole. Primarily intended for ACTIVE COMETS, "PsAng"
        #  is an indicator of the comets' gas-tail orientation in the sky (being in the
        #  anti-sunward direction) while "PsAMV" is an indicator of dust-tail orientation.
        #  Units: DEGREES
        df.at[idx, "phi0"] = eph["sunTargetPA"][0]
        df.at[idx, "phi1"] = eph["sunTargetPA"][-1]

        # Heliocentric distance
        df.at[idx, "r0"] = eph["r"][0]
        df.at[idx, "r1"] = eph["r"][-1]
        # Geocentric distance
        df.at[idx, "delta0"] = eph["delta"][0]
        df.at[idx, "delta1"] = eph["delta"][-1]
        # Solar phase angle
        df.at[idx, "alpha0"] = eph["alpha"][0]
        df.at[idx, "alpha1"] = eph["alpha"][-1]
        
        alpha_min = np.min(eph["alpha"])
        alpha_max = np.max(eph["alpha"])
        print(f"alpha_min, alpha_max = {alpha_min}, {alpha_max}")
    return df


def tab_phot(df, date, out):
    """Create obs information tex table from 15-columns DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
      15 columns NEOs DataFrame
    date : str
        date like 2024-06-16
    out : str
      output filename
    """
    
    # Like 2023 Mar 24
    # Time like 2022 Dec 20 15:30:37--16:29:49
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    # Time like 2022 December 05
    date = datetime.datetime.strftime(date, "%B %d, %Y")
    
    head = (
        r"""\begin{table*}
        \caption{\label{t7}Summary of the observations\label{tab:obs}}
        \centering
        \begin{tabular}{lrcccccccccc}
        \hline\hline
        Object & Obs. Date & $t_\mathrm{exp}$ & $N_\mathrm{img}$ & V     & $\alpha$ & $v$                & Air Mass   & Seeing   \\
               & (UTC)     & (s)              &                  & (mag) & (deg)    & (arcsec s$^{-1}$)  &            & (arcsec) \\
        \hline
        """)

    foot = (
      r"""\end{tabular}
      \tablefoot{
            Observation time in UT in midtime of exposure (Obs. Date),
            total exposure time per frame ($t_{\mathrm{exp}}$),
            and the number of images ($N_\mathrm{img}$) are listed.
            Predicted V band apparent magnitude (V), 
            phase angle ($\alpha$),
            and 
            apparent angular rate of asteroids
            at the observation starting time
            are referred to NASA Jet Propulsion Laboratory (JPL) Horizons
            """
            +
            f" as of {date}."
            +
            r"""
            Elevations of asteroids to calculate air mass range (Air Mass) are 
            also referred to NASA JPL Horizons.
            Seeing FWHM (Seeing) in the r band for Seimei observations
            measured by computing the FWHM of reference stars are also listed.
            }
            \end{table*}""")


    with open(out, "w") as f:
        f.write(head)
        for idx, row in df.iterrows():
            objtex = row["objtex"]
            
            # Not included in the table
            r = (row["r0"] + row["r1"])*0.5
            delta = (row["delta0"] + row["delta1"])*0.5
            print(f"  {idx+1}: Heliocentric distance and observer-centric distance: r={r:.3f}, delta={delta:.3f}")
            
            # Time like 2022 Dec 20 15:30:37--16:29:49
            t0 = row["obsdate0"].replace("T", " ")[:19]
            t1 = row["obsdate1"].replace("T", " ")[:19]
            # 2022-12-20
            t0_head = t0[0:10]
            t0_head = datetime.datetime.strptime(t0_head, "%Y-%m-%d")
            # 2022 Dec 20
            t0_head = datetime.datetime.strftime(t0_head, "%Y %b %d")
            t0_tail = t0[11:19]
            t1_tail = t1[11:19]

            obstimeinfo = f"{t0_head} {t0_tail}--{t1_tail}"
            # Do not show twice
            if idx > 0:
                if t0_head == t0_head0:
                    obstimeinfo = f"{t0_tail}--{t1_tail}"
                else:
                    pass

            text = (
                f"{objtex}&"
                f"{obstimeinfo}&"
                f" {row['t_exp']} & {row['Nimg']} & {row['V0']:.1f} &"
                f" {row['alpha0']:.1f} &"
                f" {row['vnorm0']:.2f} & {row['airmass_min']:.2f}--{row['airmass_max']:.2f} & {row['seeing']}"
                r"\\")
            f.write(text)
            f.write("\n")
            t0_head0 = t0_head
        f.write("\hline")
        f.write("\n")
        f.write(foot)


def tab_res(df, date, out):
    """Create result tex table.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame
    out : str
        output filename
    """

    # Like 2023 Mar 24
    # Time like 2022 Dec 20 15:30:37--16:29:49
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    # Time like 2022 December 05
    date = datetime.datetime.strftime(date, "%B %d, %Y")

    head = (
        r"""\begin{table*}
        \caption{\label{t7}Summary of observational results\label{tab:res}}
        \centering
        \begin{tabular}{ccccccc}
        \hline\hline
        Object & H  & P   & $\Delta$m        & g-r & r-i & Spectral type \\
               &    & (s) &                  &     &     &               \\
        \hline
        """)

    foot = (
      r"""\end{tabular}
      \tablefoot{
            Absolute magnitudes (H)
            are referred to NASA Jet Propulsion Laboratory (JPL) Horizons
            """
            +
            f" as of {date}."
            +
            r"""
            }
            \end{table*}""")


    with open(out, "w") as f:
        f.write(head)
        for idx, row in df.iterrows():
            obj = row["obj"]
            objtex = row["objtex"]
            H      = row["H"]

            P, Perr = row["P"], row["Perr"]
            dm, dmerr = row["dm"], row["dmerr"]
            g_r, g_rerr = row["g_r"], row["g_rerr"]
            r_i, r_ierr = row["r_i"], row["r_ierr"]

            if obj == "2021 TY14":
                stype = "X"
            else:
                stype = "S"
            
            text = (
                f"{objtex} & {H} &"
                f" ${P:.3f}\pm{Perr:.3f}$ &"
                f" ${dm:.3f}\pm{dmerr:.3f}$ &"
                f" ${g_r:.3f}\pm{g_rerr:.3f}$ &"
                f" ${r_i:.3f}\pm{r_ierr:.3f}$ &{stype}"
                r"\\")
            f.write(text)
            f.write("\n")
        f.write("\hline")
        f.write("\n")
        f.write(foot)


# Plot ========================================================================
mycolor = [
    "#AD002D", "#1e50a2", "#69821b", "#f055f0", "#afafb0", 
    "#0095b9", "#89c3eb", "#ec6800", "cyan", "gold",
    "magenta"
    ] 
mycolor = mycolor*500

# linestyle
myls = ["solid", "dashed", "dashdot", "dotted", 
        (0, (5, 3, 1, 3, 1, 3)), (0, (4,2,1,2,1,2,1,2)),
        (0, (4,2,1,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2,1,2,1,2)),
        ]
myls = myls*100

# marker
mymark = ["o", "^", "s", "D", "*", "v", "<", ">", "h", "x"]
mymark = mymark*500

bandcolor = {
    "g": "#69821b", "r": "#AD002D", "i": "magenta", "z": "#9400d3"}
bandmark = {
    "g": mymark[0], "r": mymark[1], "i": mymark[2], "z": mymark[3]}


import matplotlib as mpl
def plot_jbstyle():
    """
    Set matplotlib style for beautiful figures and math text,
    including fonts, figure size, axes, ticks, lines, legend, and math.
    """
    import matplotlib as mpl
    from matplotlib import _mathtext as mathtext

    # ---------- Font ----------
    # Use Helvetica if available; otherwise, use Arial
    mpl.rcParams.update({
        "font.family": "sans-serif",
        #"font.sans-serif": ["Helvetica"],
        "font.sans-serif": ["Arial"],
        "font.weight": 550,
        "font.size": 14,
    })

    # ---------- Figure size ----------
    # Approximate two-column width ≈ 7.2 inch
    mpl.rcParams.update({
        "figure.figsize": (7.2, 4.8),
        "figure.dpi": 300,
    })

    # ---------- Axes ----------
    mpl.rcParams.update({
        "axes.linewidth": 1.4,
        "axes.labelsize": 18,
        "axes.labelweight": 600,
        "axes.titlesize": 20,
        "axes.titleweight": 550,
    })

    # ---------- Ticks ----------
    mpl.rcParams.update({
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    # ---------- Lines ----------
    mpl.rcParams.update({
        "lines.linewidth": 1.5,
    })

    # ---------- Legend ----------
    mpl.rcParams.update({
        "legend.frameon": False,
        "legend.fontsize": 12,
    })

    # ---------- Math text ----------
    # Use Computer Modern font for math text
    mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    # Space after the sub/superscript
    mathtext.FontConstantsBase.script_space = 0.01
    # Space between the string and sub/superscript
    mathtext.FontConstantsBase.delta = 0.01
    # Align subscript to baseline
    mathtext.FontConstantsBase.subdrop = 0
    mathtext.FontConstantsBase.sub1 = 0
    # Scale of sub/superscripts
    mathtext.SHRINK_FACTOR = 0.6


def stype2colmark(stype):
    """Return marker for input spectral type.

    Parameter
    ---------
    stype : str
        spectral type

    Returns
    -------
    color : str
        color
    marker : str
        marker
    """
    if stype=="S":
        color = mycolor[0]
        mark = mymark[0]
    elif stype=="V":
        color = mycolor[3]
        mark = mymark[1]
    elif stype=="X":
        color = mycolor[2]
        mark = mymark[2]
    elif stype=="K":
        color = mycolor[7]
        mark = mymark[3]
    elif stype=="L":
        color = mycolor[9]
        mark = mymark[5]
    elif stype=="C":
        color = mycolor[1]
        mark = mymark[6]
    elif stype=="B":
        color = mycolor[5]
        mark = mymark[7]
    elif stype=="D":
        color = mycolor[6]
        mark = "p"
    elif stype=="A":
        color = mycolor[4]
        mark = mymark[8]
    # No Q-type in SDSSMOC
    else:
        color, mark = "black", "o"
    return color, mark


def timelabel(ax, ttype):
    """

    Parameters
    ----------
    ax : matplotlib.axes
        axis of the plot
    ttype : str
        hour, day, or month

    Return
    ------
    ax : matplotlib.axes
        updated axis of the plot
    """
    if ttype=="hour":
        ax.xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 1)))
        ax.xaxis.set_major_formatter(DateFormatter('%b-%d %H:%M'))
    if ttype=="day":
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b-%d'))
    if ttype=="month":
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b-%d'))
    return ax
# Plot ========================================================================


# Error =======================================================================
def log10err(val, err):
    """Calculate log10 error.
    """
    return err/val/np.log(10)


def diverr_series(val1, err1, val2, err2):
    """Calculate error for division.

    Parameters
    ----------
    val1 : float
        value 1
    err1 : float
        error 1
    val2 : float
        value 2
    err2 : float
        error 2

    Return
    ------
    err : float
        propageted error
    """
    err_list = []
    for v1, e1, v2, e2 in zip(val1, err1, val2, err2):
        err = np.sqrt((e1*1.0/v2)**2 + (e2*v1/v2**2)**2)
        err_list.append(err)
    return err_list


def mulerr(val1, err1, val2, err2):
    """Calculate error for multiple.

    Parameters
    ----------
    val1 : float
        value 1
    err1 : float
        error 1
    val2 : float
        value 2
    err2 : float
        error 2
    """
    return np.sqrt((val2*err1)**2 + (val1*err2)**2)


def adderr(*args):
    """Calculate additional error.

    Parameters
    ----------
    args : array-like
        list of values

    Return
    ------
    err : float
        calculated error
    """
    err = np.sqrt(np.sum(np.square(args)))
    return err


def adderr_series(*args):
    """Add error of multiple pandas.Series.

    Parameters
    ----------
    args : array-like
        list of pandas.Series 

    Return
    ------
    err_s : pandas.Series
        single pandas.Series of calculated errors
    """ 
    for i,x in enumerate(args):
        assert type(x)==type(pd.Series()), "Sould be Series"
        if i==0:
            temp = x.map(np.square)
        else:
            temp += x.map(np.square)
    err_s = temp.map(np.sqrt)
    return err_s
# Error =======================================================================


# Lightcurves =================================================================
def Zappala1990(stype):
    """
    Obtain s parameter in Zappala+1990.

    Parameter
    ---------
    stype : str
        spectral type

    Return
    ------
    s : float
        s parameter
    """
    if stype == "S":
        s = 0.030
    elif stype == "C":
        s = 0.015
    elif stype == "M":
        s = 0.013
    elif stype == "X":
        s = 0.013
    else:
        assert False, "Check"
    return s


# Both lightcurves (2 rotations) and phased lightcurves
def plot_lc(
        df,
        key_t,
        keys_mag,
        keys_magerr,
        rotP_sec,
        outpath="lc.jpg",
        figsize=(16, 4),
        relative=False,
        bandcolor=None,
        bandmark=None,
        t0_lc=None,
        offset_relative=0.5,
        ymargin=0.4
    ):

    # ---- time in seconds from zero ----
    t = (df[key_t] - df[key_t].min()) * 24. * 3600.
    phase = (t / rotP_sec) % 1

    fig = plt.figure(figsize=figsize)
    ax_time  = fig.add_axes([0.05, 0.15, 0.44, 0.75])
    ax_phase = fig.add_axes([0.60, 0.15, 0.22, 0.75])

    if t0_lc is None:
        t0_lc = t.min()
    t1_lc = t0_lc + 2 * rotP_sec

    y_all = []

    n_bands = len(keys_mag)

    # ---- plot for each band ----
    for i, (k_mag, k_err) in enumerate(zip(keys_mag, keys_magerr)):

        y  = df[k_mag].values
        ye = df[k_err].values

        ok = np.isfinite(y) & np.isfinite(ye)
        if not np.any(ok):
            continue

        t_b, y_b, ye_b = t[ok], y[ok], ye[ok]
        ph_b = phase[ok]

        # relative magnitude
        if relative:
            y_b = y_b - np.mean(y_b)
            # apply offset for 3 bands
            if n_bands == 3:
                if i == 0:
                    y_b -= offset_relative
                elif i == 2:
                    y_b += offset_relative

        col  = bandcolor.get(k_mag[0], "black") if bandcolor else "black"
        mark = bandmark.get(k_mag[0], "o") if bandmark else "o"

        # ---- TIME PLOT: cut to [t0_lc, t0_lc+2P] ----
        inwin = (t_b >= t0_lc) & (t_b <= t1_lc)
        t_plot, y_plot, ye_plot = t_b[inwin], y_b[inwin], ye_b[inwin]

        ax_time.errorbar(
            t_plot, y_plot, yerr=ye_plot,
            fmt=mark, markersize=6,
            markerfacecolor='none', markeredgecolor=col,
            ecolor=col, elinewidth=0.7, capsize=0,
            linestyle='none', label=k_mag
        )

        # ---- PHASE PLOT WITH WRAP (-0.2 to 1.2) ----
        # base
        ax_phase.errorbar(
            ph_b, y_b, yerr=ye_b, fmt=mark, markersize=6,
            markerfacecolor='none', markeredgecolor=col,
            ecolor=col, elinewidth=0.7, capsize=0,
            linestyle='none'
        )
        
        # shifted (phase - 1) for range -0.2 to 0
        ph_b_m1 = ph_b - 1
        ok_m1 = (ph_b_m1 >= -0.2) & (ph_b_m1 <= 1.2)
        ax_phase.errorbar(
            ph_b_m1[ok_m1], y_b[ok_m1], yerr=ye_b[ok_m1], fmt=mark, markersize=6,
            markerfacecolor='none', markeredgecolor=col,
            ecolor=col, elinewidth=0.7, capsize=0,
            linestyle='none'
        )
        
        # shifted (phase + 1) for range 1 to 1.2
        ph_b_p1 = ph_b + 1
        ok_p1 = (ph_b_p1 >= -0.2) & (ph_b_p1 <= 1.2)
        ax_phase.errorbar(
            ph_b_p1[ok_p1], y_b[ok_p1], yerr=ye_b[ok_p1], fmt=mark, markersize=6,
            markerfacecolor='none', markeredgecolor=col,
            ecolor=col, elinewidth=0.7, capsize=0,
            linestyle='none'
        )

        y_all.append(y_b)

    # ---- labels ----
    ylabel = "Relative magnitude" if relative else "Observed magnitude"
    ax_time.set_xlabel("Elapsed time [s]")
    ax_time.set_ylabel(ylabel)
    ax_phase.set_xlabel("Rotation phase")
    ax_phase.set_ylabel(ylabel)
    ax_phase.set_xlim(-0.2, 1.2)

    # ---- y-limits with margin ----
    if y_all:
        y_min = min([y.min() for y in y_all]) - ymargin
        y_max = max([y.max() for y in y_all]) + ymargin
        ax_time.set_ylim(y_max, y_min)   # invert y-axis
        ax_phase.set_ylim(y_max, y_min)  # match phase plot

    ax_time.legend(fontsize=8)
    ax_phase.legend(fontsize=8)

    # ---- secondary axis (time) ----
    sec_ax = ax_phase.secondary_xaxis(
        "top",
        functions=(lambda p: p * rotP_sec, lambda tt: tt / rotP_sec)
    )
    sec_ax.set_xlabel("Time [s]")

    # ---- TIME RANGE ----
    xmargin = 0.05 * (2 * rotP_sec)
    ax_time.set_xlim([t0_lc - xmargin, t1_lc + xmargin])

    # ---- save figure ----
    fig.savefig(outpath)
    print(f"-> saved: {outpath}")
    plt.close()
    return fig


def plot_plc(
        df,
        key_t,
        keys_mag,
        keys_magerr,
        rotP_sec,
        rotPerr_sec,
        outpath="plc.jpg",
        relative=False,
        offset_relative=1.0,
        bandcolor=None,
        bandmark=None,
        ymargin=0.5,
        objtex=None,
        ylim=None,
    ):
    """
    Phase-folded light curve only.
    """

    # ---- time in seconds from zero ----
    t = (df[key_t] - df[key_t].min()) * 24. * 3600.
    phase = (t / rotP_sec) % 1

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.20, 0.18, 0.75, 0.68])

    y_all = []
    n_bands = len(keys_mag)

    # ---- plot each band ----
    for i, (k_mag, k_err) in enumerate(zip(keys_mag, keys_magerr)):

        y  = df[k_mag].values
        ye = df[k_err].values
        ok = np.isfinite(y) & np.isfinite(ye)
        if not np.any(ok):
            continue

        ph_b = phase[ok]
        y_b  = y[ok]
        ye_b = ye[ok]

        # relative magnitude
        if relative:
            y_b = y_b - np.mean(y_b)
            if n_bands == 3:
                if i == 0:
                    y_b -= offset_relative
                    y_text = -offset_relative
                elif i == 1:
                    y_text = 0
                elif i == 2:
                    y_b += offset_relative
                    y_text = +offset_relative

        col  = bandcolor.get(k_mag[0], "black") if bandcolor else "black"
        mark = bandmark.get(k_mag[0], "o") if bandmark else "o"

        # Phase 0–1
        ax.errorbar(
            ph_b, y_b, yerr=ye_b,
            fmt=mark, markersize=6,
            markerfacecolor="none", markeredgecolor=col,
            ecolor=col, elinewidth=0.7, capsize=0,
            linestyle="none", label=None,
        )
        # Put mag
        ax.text(
            -0.27, y_text, k_mag[0], color=col, size=20, 
            horizontalalignment="center")

        ph_m1 = ph_b - 1
        ok_m1 = (ph_m1 >= -0.2) & (ph_m1 <= 1.2)
        if np.any(ok_m1):
            ax.errorbar(
                ph_m1[ok_m1], y_b[ok_m1], yerr=ye_b[ok_m1],
                fmt=mark, markersize=6,
                markerfacecolor="none", markeredgecolor=col,
                ecolor=col, elinewidth=0.7, capsize=0,
                linestyle="none"
            )

        ph_p1 = ph_b + 1
        ok_p1 = (ph_p1 >= -0.2) & (ph_p1 <= 1.2)
        if np.any(ok_p1):
            ax.errorbar(
                ph_p1[ok_p1], y_b[ok_p1], yerr=ye_b[ok_p1],
                fmt=mark, markersize=6,
                markerfacecolor="none", markeredgecolor=col,
                ecolor=col, elinewidth=0.7, capsize=0,
                linestyle="none"
            )

        y_all.append(y_b)

    # ---- labels ----
    ylabel = "Relative magnitude" if relative else "Observed magnitude"
    ax.set_xlabel("Rotation phase")
    ax.set_ylabel(ylabel)

    # Put period
    ax.text(
        0.5, 0.05, f"$P={rotP_sec:.3f}\pm{rotPerr_sec:.3f}$ s", color="black", size=22, 
        horizontalalignment="center", transform=ax.transAxes)

    # Put object name
    if objtex:
        ax.text(
            0.5, 0.91, objtex, color="black", size=22, 
            horizontalalignment="center", transform=ax.transAxes)

    # ---- limits ----
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks(np.arange(-0.2, 1.21, 0.2))
    ax.set_xlabel("Rotation phase")


    if ylim is not None:
        y0 = np.min(ylim)
        y1 = np.max(ylim)
        ax.set_ylim(y1, y0)
    elif y_all:
        y_min = min([yy.min() for yy in y_all]) - ymargin
        y_max = max([yy.max() for yy in y_all]) + ymargin
        ax.set_ylim(y_max, y_min)  # invert y-axis

    # Vertical lines at rotation phases of 0 and 1
    #ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    #ax.axvline(1, color='gray', linestyle='--', linewidth=1)

    # ---- secondary axis (time) ----
    sec_ax = ax.secondary_xaxis(
        "top",
        functions=(lambda p: p * rotP_sec, lambda tt: tt / rotP_sec)
    )
    # Every 5 s 
    sec_ax.set_xticks(np.arange(0, rotP_sec * 1.21, 5))
    sec_ax.set_xlabel("Time [s]")

    # ---- save ----
    fig.savefig(outpath)
    print(f"-> saved: {outpath}")
    plt.close()
    return fig



def plot_lc_full(
        df,
        key_t,
        keys_mag,
        keys_magerr,
        outpath="lc_full.jpg",
        figsize=(16, 4),
        relative=False,
        bandcolor=None,
        bandmark=None,
        offset_relative=0.5,
        ymargin=0.3
    ):
    """
    Full lightcurve plot for appendix.
    Only time plot, phase plot omitted.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- time in seconds from zero ----
    t = (df[key_t] - df[key_t].min()) * 24. * 3600.

    fig = plt.figure(figsize=figsize)
    ax_time = fig.add_axes([0.10, 0.20, 0.87, 0.75])  

    y_all = []

    n_bands = len(keys_mag)

    for i, (k_mag, k_err) in enumerate(zip(keys_mag, keys_magerr)):

        y  = df[k_mag].values
        ye = df[k_err].values

        ok = np.isfinite(y) & np.isfinite(ye)
        if not np.any(ok):
            continue

        t_b, y_b, ye_b = t[ok], y[ok], ye[ok]

        # relative magnitude
        if relative:
            y_b = y_b - np.mean(y_b)
            if n_bands == 3:
                if i == 0:
                    y_b -= offset_relative
                    y_text = -offset_relative
                elif i == 2:
                    y_b += offset_relative

        col  = bandcolor.get(k_mag[0], "black") if bandcolor else "black"
        mark = bandmark.get(k_mag[0], "o") if bandmark else "o"
        
        #label = f"{k_mag[0]}"
        label = None

        ax_time.errorbar(
            t_b, y_b, yerr=ye_b,
            fmt=mark, markersize=6,
            markerfacecolor='none', markeredgecolor=col,
            ecolor=col, elinewidth=0.7, capsize=0,
            linestyle='none', label=label
        )

        y_all.append(y_b)


    # ---- labels ----
    ylabel = "Relative magnitude" if relative else "Observed magnitude"
    ax_time.set_xlabel("Elapsed time [s]")
    ax_time.set_ylabel(ylabel)

    # ---- y-limits with margin ----
    if y_all:
        y_min = min([y.min() for y in y_all]) - ymargin
        y_max = max([y.max() for y in y_all]) + ymargin
        ax_time.set_ylim(y_max, y_min)  # invert y-axis

    # Put mag on the left
    for i, (k_mag, k_err) in enumerate(zip(keys_mag, keys_magerr)):
        if relative:
            y_b = y_b - np.mean(y_b)
            if n_bands == 3:
                if i == 0:
                    y_text = -offset_relative
                elif i == 2:
                    y_text = +offset_relative
                else:
                    y_text = 0

        col  = bandcolor.get(k_mag[0], "black") if bandcolor else "black"
        xmin, xmax = ax_time.get_xlim()
        x_text = xmin + 0.02*(xmax-xmin)
        ax_time.text(
            x_text, y_text, k_mag[0], color=col, size=20, 
            horizontalalignment="center")

    ax_time.legend(loc="upper right")

    # ---- save figure ----
    fig.savefig(outpath)
    print(f"-> saved: {outpath}")
    plt.close()
    return fig

def ls_period_search(
        t, mag, magerr,
        outpath=None,
        nterm=1,
        norm="standard",
        N_freq=10000,
        order=5,
        Pth=0.2,
        prob=(0.1, 0.01, 0.001),
        figsize=(8,8),
        obj_label="",
        colors=None,
        linestyles=None,
    ):
    """
    Find strongest Lomb–Scargle peak and plot power spectrum.

    Parameters
    ----------
    t, mag, magerr : array
        time [days or arbitrary], magnitude, magnitude error
        (only t is used)
    outpath : str or None
        if given, save figure
    nterm : int
        number of harmonics
    prob : tuple
        false alarm level probabilities

    Returns
    -------
    P_s : float
        best period [sec]
    fap : float
        false alarm probability at peak
    """

    # 1) Time to second
    T0 = np.min(t)
    t_sec = (t - T0) * 24.*3600.

    # 2) Frequency range (2 sec to full arc)
    arc_s = np.max(t_sec)
    Pmin, Pmax = 2.0, arc_s
    f_min, f_max = 1/Pmax, 1/Pmin

    print(f"  Observation arc: {arc_s:.1f} s")
    print(f"  Search range {Pmin:.2f}–{Pmax:.2f} s")
    print(f"  Number of harmonics = {nterm}")

    # 3) Grid
    freq = np.linspace(f_min, f_max, N_freq)

    # 4) LS
    ls = LombScargle(
        t_sec, mag, normalization=norm, nterms=nterm, center_data=True
    )
    power = ls.power(freq)

    ## Peak search
    peak_idx = argrelmax(power, order=order)[0]
    peak_power = power[peak_idx]
    peak_freq = freq[peak_idx]

    # Get top
    sort = np.argsort(peak_power)[::-1]
    peak_sorted = peak_power[sort]
    freq_sorted = peak_freq[sort]

    # Best above threshold
    mask = peak_sorted > Pth
    if not np.any(mask):
        print("  No peak above threshold.")
        P_s = None
        fap = None
    else:
        P_s = 1./freq_sorted[mask][0]
        best_power = peak_sorted[mask][0]
        fap = ls.false_alarm_probability(best_power)
        print(f"    best peak: P = {P_s:.3f} s, power={best_power:.2f}")

    # 6) Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.14, 0.80, 0.80])

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized LS power")

    ax.plot(freq, power, lw=0.7, color="black")

    # False alarm levels
    fal = ls.false_alarm_level(prob)

    # default cosmetics
    if colors is None:
        colors = ["red", "green", "blue"]
    if linestyles is None:
        linestyles = ["--", "-.", ":"]

    for p, f, c, ls_ in zip(prob, fal, colors, linestyles):
        label = f"{((1-p)*100):.1f} %"
        ax.axhline(f, color=c, ls=ls_, lw=1, label=label)

    ax.set_ylim(0, 0.5)
    ax.set_xlim(f_min, f_max)
    ax.legend()

    if outpath:
        fig.savefig(outpath)
    plt.close()

    return P_s, fap, freq, power



def ls_mc_plot(
        t, mag, magerr,
        rotP_s,
        N_mc=200,
        nterm=1,
        norm="standard",
        outpath=None,
        bins=20,
        figsize=(8, 8),
    ):
    """
    Monte-Carlo Lomb-Scargle to get period and amplitude,
    and make P vs dmag plot with hist attached on top/right.

    Parameters
    ----------
    t : array-like
        time series (seconds)
    mag : array-like
        magnitude time series
    magerr : array-like
        magnitude uncertainty time series
    rotP_s : float
        expected rotation period [s], used to define search range
    N_mc : int
        number of Monte-Carlo trials
    nterm : int
        number of harmonic terms for LS model
    norm : str
        LS normalization (e.g. "psd", "standard")
    outpath : str or None
        if given, save figure to this file path
    bins : int
        histogram bins
    figsize : tuple
        figure size
    """

    # 1) Time to second
    T0 = np.min(t)
    t_sec = (t - T0) * 24.*3600.

    # ---- frequency range ----
    # This is very important to estimate parameters
    P0 = rotP_s * 0.95
    P1 = rotP_s * 1.05
    f_min, f_max = 1/P1, 1/P0

    N_freq = 3000
    freq = np.linspace(f_min, f_max, N_freq)

    # ---- run MC ----
    P_list, dm_list = [], []

    for n in range(N_mc):
        if n == 0:
            y = mag
        else:
            y = np.random.normal(mag, magerr, len(mag))

        ls = LombScargle(
            t_sec, y, normalization=norm,
            nterms=nterm, center_data=False,
        )
        power = ls.power(freq)
        idx = np.argmax(power)

        peak_f = freq[idx]
        peak_P = 1. / peak_f
        P_list.append(peak_P)

        # --- amplitude ---
        params = ls.model_parameters(peak_f)
        x_model = np.linspace(0, 1, 200)

        y_model = params[0]
        for i in range(nterm):
            y_model += (
                params[2*i+1] * np.sin(2*np.pi*x_model*(i+1)) +
                params[2*i+2] * np.cos(2*np.pi*x_model*(i+1))
            )

        dm = y_model.max() - y_model.min()
        dm_list.append(dm)
       
        # For debug to check model curves =====================================
        #fig = plt.figure(figsize=figsize)
        #ax = fig.add_axes([0.15, 0.14, 0.80, 0.74])
        #ax.set_xlabel("Time s ")
        #ax.set_ylabel("Mag")
        #ax.scatter(t_sec, mag, lw=0.7, color="black")
        #t_model = x_model * peak_P
        #ax.plot(t_model, y_model, lw=0.7, color="black")
        #ax.set_xlim([0, 100])
        #plt.show()
        # For debug to check model curves =====================================


    # ---- statistics ----
    P_mean, P_std = np.mean(P_list), np.std(P_list)
    dm_mean, dm_std = np.mean(dm_list), np.std(dm_list)

    # ---- figure ----
    fig = plt.figure(figsize=figsize)

    main = fig.add_axes([0.15, 0.15, 0.65, 0.65])
    top  = fig.add_axes([0.15, 0.80, 0.65, 0.12], sharex=main)
    side = fig.add_axes([0.80, 0.15, 0.12, 0.65], sharey=main)

    # main plot
    main.scatter(P_list, dm_list, s=1, color="black")
    main.set_xlabel("Period [s]")
    main.set_ylabel("Lightcurve amplitude [mag]")
    
    # Add 1-sigma line
    xy = np.vstack([P_list, dm_list])
    kde = gaussian_kde(xy)
    xmin, xmax = np.min(P_list), np.max(P_list)
    ymin, ymax = np.min(dm_list), np.max(dm_list)
    # create grid
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 200),
                       np.linspace(ymin, ymax, 200))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    # 1-sigma: find contour level that encloses ~68% of probability
    # sort density values descending
    Z_sorted = np.sort(Z.ravel())[::-1]
    Z_cumsum = np.cumsum(Z_sorted)
    Z_cumsum /= Z_cumsum[-1]  # normalize to 1
    # find density threshold for 68%
    level = Z_sorted[Z_cumsum <= 0.6827][-1]
    
    # plot contour
    main.contour(
        X, Y, Z, levels=[level], colors='black', linestyles='--', linewidths=1, zorder=10)

    # histogram (normalize)
    top.hist(P_list, bins=bins, density=True,
             histtype="step", color="black")
    side.hist(dm_list, bins=bins, density=True,
              histtype="step", color="black",
              orientation='horizontal')

    # top histogram: vertical dashed lines at P_mean ± P_std
    top.axvline(P_mean - P_std, color='black', linestyle='--', lw=1)
    top.axvline(P_mean + P_std, color='black', linestyle='--', lw=1)
    
    # side histogram: horizontal dashed lines at dm_mean ± dm_std
    side.axhline(dm_mean - dm_std, color='black', linestyle='--', lw=1)
    side.axhline(dm_mean + dm_std, color='black', linestyle='--', lw=1)

    # cleaning
    top.tick_params(labelbottom=False)
    side.tick_params(labelleft=False)
    top.set_yticks([])
    side.set_xticks([])

    # stats text
    text = (
        fr"$P = {P_mean:.3f}\pm{P_std:.3f}$ s" + "\n" +
        fr"$\Delta m = {dm_mean:.3f}\pm{dm_std:.3f}$"
    )

    main.text(
        0.05, 0.95, text,
        transform=main.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )


    xmargin = 5*P_std
    ymargin = 5*dm_std
    
    xmin_plot = P_mean - xmargin
    xmax_plot = P_mean + xmargin
    ymin_plot = dm_mean - ymargin
    ymax_plot = dm_mean + ymargin
    
    main.set_xlim(xmin_plot, xmax_plot)
    main.set_ylim(ymin_plot, ymax_plot)
    
    top.set_xlim(xmin_plot, xmax_plot)
    side.set_ylim(ymin_plot, ymax_plot)


    if outpath:
        fig.savefig(outpath)

    plt.close()

    return P_mean, P_std, dm_mean, dm_std, np.array(P_list), np.array(dm_list)
# Lightcurves =================================================================


# Colors ======================================================================
# Tonry2012
def gr2V_mag(g, gerr, r, rerr):
    """Convert g and r mag to V-band magnitude based on Tonry+2012.

    Parameters
    ----------
    g : float
        g-band magnitude
    gerr : float
        g-band magnitude error
    r : float
        r-band magnitude
    rerr : float
        r-band magnitude error

    Returns
    -------
    V : float
        V-band magnitude
    Verr : float
        V-band magnitude error
    """
    # Equation (2) in block 4 in table 6
    V = r + 0.006 + 0.474*(g-r)
    # Error of colors
    grerr = adderr(gerr, rerr)
    # rerr, error of conversion, and error of colors
    Verr = adderr(rerr, 0.012, 0.474*grerr)
    return V, Verr


def SDSS2PS_mag(df, key0=None, key1=None):
    """Convert SDSS magnitude to Pan-STARRS magnitude.

    (convert origianl 'g' to 'g_SDSS', and add converted magnitude as 'g_PS')
    See block 1 in table 6 in Tonry+2012.
    Note: 'err's are error for conversion itself.
           ex) when g1 = g0 + (g0-r0)*C, and conversion error is err, 
               total error (err_total) is 
                 err_total**2 = err**2 + (C*g0err)**2 + (C*r0err)**2

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe contains SDSS g,r,i,z-band magnitude
    key0 : dict, optional
        magnitude and magnitude error keys of original DataFrame
    key1 : dict, optional
        magnitude and magnitude error keys to be added

    Return
    ------
    df : pandas.DataFrame
        dataframe contains Pan-STARRS g,r,i,z-band magnitude
    """
    # Data dimension
    n = len(df)

    # Original keys to be converted.
    if key0 is None:
      key0 = {
        "g":"g", "gerr":"gerr",
        "r":"r", "rerr":"rerr",
        "i":"i", "ierr":"ierr",
        "z":"z", "zerr":"zerr"
        }
    # New keys.
    if key1 is None:
      key1 = {
        "g":"g_PS", "gerr":"gerr_PS",
        "r":"r_PS", "rerr":"rerr_PS",
        "i":"i_PS", "ierr":"ierr_PS",
        "z":"z_PS", "zerr":"zerr_PS"
       }

    # Estimate g, r, i, and z
    # Coefficients from Table 6 in Tonry+2012
    coeff = {
        "B0":  [-0.012,  0.000,  0.004, -0.013], 
        "B1":  [-0.139, -0.007, -0.014,  0.039], 
        "Berr": [ 0.007,  0.002,  0.003,  0.009]
        }
    bands = ["g", "r", "i", "z"]
    for idx, b in enumerate(bands):
        B0 = coeff["B0"][idx]
        B1 = coeff["B1"][idx]
        err = coeff["Berr"][idx]
        # Use g-r color for all conversion (see Table 6)
        df[key1[b]] = df[key0[b]] + B0 + (df[key0["g"]] - df[key0["r"]])*B1
        print(f"  Conversion: {key1[b]} = {key0[b]} + {B0} + (g-r)*{B1}")
        ## Create a pandas.Series of a conversion error
        converr = [err]*n
        converr_series = pd.Series(data=converr)
        df[key1[f"{b}err"]] = adderr_series(
            df[key0[f"{b}err"]], converr_series, 
            B1*df[key0["gerr"]], B1*df[key0["rerr"]])
    return df


def PS2SDSS_col(df, key0=None, key1=None):
    """Convert colors in the Pan-STARRS to SDSS.

    See block 3 in table 6 in Tonry+2012.
    Note: 'err's are error for conversion itself.
          When 
          Color1 = C0 + C1*(Color0)*C , and conversion error is err1 and err2
            (two errors since the equations are some of two eqs. in Table 6)
          err_total**2 = err1**2 + err2**2 + (C*(g0-r0)err)**2

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe contains gri color
    key0 : dict, optional
        color and color error keys of original DataFrame
    key1 : dict, optional
        color and color error keys to be added

    Return
    ------
    df : pandas.DataFrame
        dataframe contains SDSS color
    """
    # Data dimension
    n = len(df)

    # Original keys to be converted.
    if key0 is None:
        key0 = {
            "g_r":"g_r_PS", "g_rerr":"g_rerr_PS",
            "r_i":"r_i_PS", "r_ierr":"r_ierr_PS",
            "i_z":"i_z_PS", "i_zerr":"i_zerr_PS",
        }
    # New keys.
    if key1 is None:
        key1 = {
            "g_r":"g_r_SDSS", "g_rerr":"g_rerr_SDSS",
            "r_i":"r_i_SDSS", "r_ierr":"r_ierr_SDSS",
            "i_z":"i_z_SDSS", "i_zerr":"i_zerr_SDSS",
        }

    # Coefficients from block 3 in Table 6 (Tonry+2012)
    coeff = {
        "B0":   [ 0.014, -0.001, -0.004,  0.013], 
        "B1":   [ 0.162,  0.011,  0.020, -0.050], 
        "Berr": [ 0.009,  0.004,  0.005,  0.010]
        }
    # Calculate (g-r)SDSS from equations (1) - (2)
    df[key1["g_r"]] = (
        (coeff["B0"][0] - coeff["B0"][1]) 
        + (coeff["B1"][0] - coeff["B1"][1] + 1)*df[key0["g_r"]]
        )
    # Sum of err[0], err[1], and (B1[0]-B1[1]+1)*g_rerr
    err1 = [coeff["Berr"][0]]*n
    err2 = [coeff["Berr"][1]]*n
    err1_s = pd.Series(data=err1)
    err2_s = pd.Series(data=err2)
    err_gr = df[key0["g_rerr"]]
    df[key1["g_rerr"]] = adderr_series(
        err1_s, err2_s, (coeff["B1"][0]-coeff["B1"][1]+1)*err_gr)

    # Calculate (r_i)SDSS from equations (2) - (3)
    df[key1["r_i"]] = (
      (coeff["B0"][1] - coeff["B0"][2]) 
      + (coeff["B1"][1] - coeff["B1"][2])*df[key0["g_r"]] + df[key0["r_i"]]
      )
    # Sum of err[1], err[2], (B1[1]-B1[2])*g_rerr, and r_ierr
    err3 = [coeff["Berr"][2]]*n
    err3_s = pd.Series(data=err3)
    err_ri = df[key0["r_ierr"]]
    df[key1["r_ierr"]] = adderr_series(
      err2_s, err3_s, (coeff["B1"][1]-coeff["B1"][2])*err_gr, err_ri)

    # Calculate (i_z)SDSS from equations (3) - (4)
    df[key1["i_z"]] = (
      (coeff["B0"][2] - coeff["B0"][3]) 
      + (coeff["B1"][2] - coeff["B1"][3])*df[key0["g_r"]] + df[key0["i_z"]]
      )
    # Sum of err[2], err[3], (B1[2]-B1[3])*g_rerr, and i_zerr
    err4 = [coeff["Berr"][3]]*n
    err4_s = pd.Series(data=err4)
    err_iz = df[key0["i_zerr"]]
    df[key1["i_zerr"]] = adderr_series(
      err3_s, err4_s, (coeff["B1"][2]-coeff["B1"][3])*err_gr, err_iz)

    return df


def calc_wmean(val, valerr):
    w = 1 / valerr**2
    mean = np.average(val, weights=w)
    err  = np.sqrt(1 / np.sum(w))
    return mean, err


def calc_rms_residual(arr1, arr2):
    """Calculate rms residual of two arrays.

    Parameters
    ----------
    arr1, arr2 : array-like
        two arrays

    Return
    ------
    rms : rms residual
    """
    resi = arr1 - arr2
    rms = np.sqrt(np.mean(resi**2)/len(resi))
    return rms
# Colors ======================================================================



# Others ======================================================================
def calc_JPLephem(asteroid, date0, date1, step, obscode, air=False):
    """Calculate asteroid ephemeris.
  
    Parameters
    ----------
    asteroid : str
      asteroid name like "Ceres", "2019 FA" (should have space)
    date0 : str
      ephemeris start date like "2020-12-12"
    date1 : str
      ephemeris end date like "2020-12-12"
    step : str
      ephemeris step date like '1d' for 1-day, '30m' for 30-minutes
    obscode : str, optional
      IAU observation code. 
      371:Okayama Astronomical Observatory
      381:Kiso observatory
    air : bool, optional
      whether consider air pressure
  
    Return
    ------
    ephem : astropy.table.table.Table
      calculated ephemeris
    """
  
    obj = Horizons(id=asteroid, location=obscode,
          epochs={'start':date0, 'stop':date1, 'step':step})
    eph = obj.ephemerides(refraction=air)
    return eph


def D_from_Hp(H, Herr, p, perr):
    """Calulate diameter from absolute magnitude H and albedo.

    Parameters
    ----------
    H, Herr : float
        absolute magnitude and its error
    p, perr : float 
        visual geometric albedo and its error

    Returns
    -------
    D, Derr : float
        diameter and its erorr in km
    """
    D = 1329.*p**(-0.5)*10**(-0.2*H)
    # H part
    errH_2 = Herr**2*(1329.*p**(-0.5)/5.*10**(-0.2*H)*np.log(10))**2
    # p part
    errp_2 = perr**2*(1329.*p**(-1.5)/2.*10**(-0.2*H))**2
    Derr = np.sqrt(errp_2 + errH_2)
    return D, Derr
# Others ======================================================================
