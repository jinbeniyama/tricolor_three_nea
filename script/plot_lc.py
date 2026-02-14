#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create lightcurves for multiple objects.

Input file: 
  - gri_mag.txt (Seimei obs., 2021 UW1, 2021 TY14)
    Columns include "obj", "t_jd_ltcor", "g_red", "gmagerr", "r_red",
    "rmagerr", "i_red", "ierr", and "alpha"
  - gri_2022GQ1_1s_N80.txt (Seimei obs., 2022 GQ1)
    Columns include "obj", "t_jd_ltcor", "flux", "fluxerr", "band"
"""
import os
from argparse import ArgumentParser as ap
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from common import (
    mymark, myls, plot_jbstyle, mycolor, Zappala1990, plot_lc, plot_plc, 
    ls_period_search, ls_mc_plot, plot_lc_full
    )


if __name__ == "__main__":
    parser = ap(description="Create lightcurves for multiple objects")
    parser.add_argument(
        "res", type=str, 
        help="Photometric results (gri_mag.txt)")
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

    # Basic settings
    bands = ["g", "r", "i"]
    bandcolor = {"g": mycolor[2], "r": mycolor[0], "i": mycolor[3]}
    bandmark = {"g": mymark[0], "r": mymark[1], "i": mymark[2]}

    # Read combined data
    df_2021 = pd.read_csv(args.res, sep=" ")

    t0_dict = {
        "2021TY14": 815,    
        "2021UW1": 300, 
        "2022GQ1": 0,     
    }

    plot_jbstyle()

    # Number of trials
    N_mc = 1000
    # LT-corrected time
    key_t = "t_jd_ltcor"

    obj_list = ["2021TY14", "2021UW1", "2022GQ1"]
    objtex_list = ["2021 TY$_{14}$", "2021 UW$_{1}$", "2022 GQ$_{1}$"]
    stype_dict = {"2021TY14": "X", "2021UW1": "S", "2022GQ1":"S"}
    alpha_dict = {"2021TY14": 23.7, "2021UW1": 24.5, "2022GQ1": 81.3}

    for idx, obj in enumerate(obj_list):
        if obj == "2022GQ1":
            df = pd.read_csv(args.res_GQ1, sep=" ")
            # Radius 
            rad_GQ1 = 10
            df = df[df["radius"] == rad_GQ1]
            is_relative = True
            # Highest SNR for 2022GQ1
            key_mag, key_magerr = "imag", "imagerr"
        else:
            df = df_2021[df_2021["obj"] == obj].copy()
            is_relative = False
            # Highest SNR for both 2021TY14 and 2021UW1
            key_mag, key_magerr = "rmag", "rmagerr"

        nterm_mc = 2

        keys_mag = ["gmag", "rmag", "imag"] 
        keys_magerr = ["gmagerr", "rmagerr", "imagerr"] 
        t0_lc = t0_dict[obj]
        print(f"Processing {obj} ...")
        
        # Object name for label
        objtex = objtex_list[idx]

       # Periodic analysis ====================================================
        out = f"{obj}_LS.{args.outtype}"
        out = os.path.join(args.outdir, out)

        P_s, fap, freq, power = ls_period_search(
            t=df[key_t].values,
            mag=df[key_mag].values,
            magerr=df[key_magerr].values,
            nterm=1,
            outpath=out,
            obj_label=objtex,
            colors=mycolor[:3],
        )
        print()

       # Periodic analysis with MC ============================================
        rotP = 2*P_s
        out = f"{obj}_LS_mc.{args.outtype}"
        out = os.path.join(args.outdir, out)
        P_mean, P_std, dm_mean, dm_std, Plist, dmlist = ls_mc_plot(
            t=df[key_t].values,
            mag=df[key_mag].values,
            magerr=df[key_magerr].values,
            rotP_s=rotP,
            N_mc=N_mc,
            nterm=nterm_mc,
            outpath=out,
        )
        print(f"  P  = {P_mean:.3f}+-{P_std:.3f} s")
        print(f"  dm = {dm_mean:.3f}+-{dm_std:.3f}")

        # Calculate a/b ratio
        alpha = alpha_dict[obj]
        for stype in ["S", "C", "M"]:
            m_param = Zappala1990(stype)
            dm_alpha0 = dm_mean/(1+m_param*alpha)
            dmerr_alpha0 = dm_std/(1+m_param*alpha)
            ab_min = 10**(0.4*dm_alpha0)
            print(f"  dm = {dm_alpha0:.2f}+-{dmerr_alpha0:.2f} (at opposition, assuming {stype}-type)")
            print(f"  -> a/b >= {ab_min:.2f}")
        print()

       # Plot phased lightcurves ==============================================
        out = os.path.join(outdir, f"{obj}_plc.{args.outtype}")
        is_relative = True
        plot_plc(
            df,
            key_t,
            keys_mag,
            keys_magerr,
            P_mean,
            P_std,
            outpath=out,
            relative=is_relative,
            offset_relative=1.0,
            ylim=[-2.4, 2.4],
            bandcolor=bandcolor,
            bandmark=bandmark,
            objtex=objtex,
        )
        print()

       # Plot full lc =========================================================
        out = os.path.join(outdir, f"{obj}_lc_full.{args.outtype}")
        plot_lc_full(
            df,
            key_t,
            keys_mag,
            keys_magerr,
            relative=is_relative,
            bandcolor=bandcolor,
            bandmark=bandmark,
            outpath=out
        )
        print()
