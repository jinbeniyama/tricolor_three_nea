#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot colors of asteroids on color-color diagram.
"""
import os 
from argparse import ArgumentParser as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

from common import (
    mycolor, color_nea,  mymark, stype2colmark, SDSS2PS_mag, 
    plot_jbstyle, calc_wmean)
from hoya.core import extract_hoya, DBPATH


if __name__ == "__main__":
    parser = ap(description="Plot color-color diagram.")
    parser.add_argument(
        "res", type=str,
        help="Results with colors")
    parser.add_argument(
        "--magtype", type=str, default="PS",
        help="Magnitude system")
    parser.add_argument(
        "--outtype", type=str, default="pdf",
        help="format of output figures")
    parser.add_argument(
        "--outdir", type=str, default="fig",
        help="output directory")
    args = parser.parse_args()
    
    if not os.path.isdir(args.outdir):
      os.makedirs(args.outdir)

    col = ["obj", "stype", "H"]
    magtype = args.magtype
    stype_use = ["S", "V", "X", "K", "L", "C", "B", "D", "A"]


    # Read SDSS MOC data ======================================================
    df_S21 = extract_hoya(DBPATH, "Sergeyev2021")
    # Select high possibility objects
    p_th = 0.8
    df_S21 = df_S21[df_S21["p_stype"].astype(float) > p_th]
    print(f"N_SDSS = {len(df_S21)} (p > {p_th})")

    if magtype=="PS":
        # Change system to PS system
        print("Photometric results are in the Pan-STARRS.")
        print("Convert colors in the SDSS to those in PS.")
        df_S21 = SDSS2PS_mag(df_S21)

        # Rename
        df_S21 = df_S21.rename(
          columns={"g":"g_temp", "gerr":"gerr_temp", 
                   "r":"r_temp", "rerr":"rerr_temp", 
                   "i":"i_temp", "ierr":"ierr_temp", 
                   "z":"z_temp", "zerr":"zerr_temp"})
        df_S21 = df_S21.rename(
          columns={"g_PS":"g", "gerr_PS":"gerr", 
                   "r_PS":"r", "rerr_PS":"rerr", 
                   "i_PS":"i", "ierr_PS":"ierr", 
                   "z_PS":"z", "zerr_PS":"zerr"})
    # Read SDSS MOC data ======================================================

    # Setting for plot ========================================================
    s_SDSS, lw_SDSS = 50, 1
    s_scatter, lw_scatter = 1000, 2
    s_eb, lw_eb = 30, 2
    if magtype == "PS":
        mgtp = "Pan-STARRS"
    elif magtype == "SDSS":
        mgtp = "SDSS"
    plot_jbstyle()
    # Setting for plot ========================================================


    # g-r vs. r-i =============================================================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    ax.set_ylabel(f"$r-i$  ({mgtp})")
    ax.set_xlabel(f"$g-r$  ({mgtp})")

    # Set filters width = 0.6
    c1min, c1max = 0.3, 0.8
    c2min, c2max = -0.05, 0.45
    print(f"Color range")
    print(f"    c1 = {c1min}--{c1max}")
    print(f"    c2 = {c2min}--{c2max}")

    ax.set_xlim([c1min, c1max])
    ax.set_ylim([c2min, c2max])
    
    # Plot objects in SDSSMOC
    for idx, stype in enumerate(stype_use):
        df_type = df_S21[df_S21["stype"]==stype]
        df_type = df_type.reset_index(drop=True)
        print(f"  {stype}-complex N={len(df_type)}")
        if len(df_type) == 0:
            continue
        col, mark = stype2colmark(stype)
    
        ax.scatter(
            df_type["g"]-df_type["r"],
            df_type["r"]-df_type["i"],
            color=col, s=s_SDSS, lw=lw_SDSS, marker=mark, facecolor="None",
            edgecolor=col, zorder=-1, label=f"{stype}-complex")

    # Seimei ==================================================================
    # Read colors
    df = pd.read_csv(args.res, sep=" ")
    
    obj_list = ["2021TY14", "2021UW1", "2022GQ1"]
    objtex_list = ["2021 TY$_{14}$", "2021 UW$_1$", "2022 GQ$_1$"]

    for idx, obj in enumerate(obj_list):
        objtex = objtex_list[idx]
        df_obj = df[df["obj"] == obj]
        
        # This is measured on jupiter notebook
        if obj == "2022GQ1":
            gr, grerr, ri, rierr = color_nea(obj)
        else:
            gr, grerr = calc_wmean(df_obj["g_r"], df_obj["g_rerr"])
            ri, rierr = calc_wmean(df_obj["r_i"], df_obj["r_ierr"])

        print(f"Colors of {obj}")
        print(f"  $g-r = {gr:.3f}\pm{grerr:.3f}$")
        print(f"  $r-i = {ri:.3f}\pm{rierr:.3f}$")
        print()

        mark = "*"
        col = mycolor[idx]

        ax.errorbar(
            gr, ri, xerr=grerr, yerr=rierr,
            fmt=mark,
            markerfacecolor=col, 
            markeredgecolor="black",  
            ecolor='black', 
            ms=20,
            lw=2,
            zorder=201, 
            label=objtex,
            )
    # Seimei ==================================================================
    leg = ax.legend(loc="upper right", frameon=True)
    frame = leg.get_frame()

    frame.set_edgecolor('black')
    frame.set_linewidth(1.5)
    frame.set_alpha(1.0)


    out = f"cc_{magtype}.{args.outtype}"
    out = os.path.join(args.outdir, out)
    fig.savefig(out)
    plt.close()
    # g-r vs. r-i =============================================================
