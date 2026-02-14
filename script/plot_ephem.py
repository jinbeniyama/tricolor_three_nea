#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create ephemeris plots for multiple objects (2021 TY14, 2021 UW1, 2022 GQ1).
"""
import os
import datetime
from argparse import ArgumentParser as ap
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DateFormatter, DayLocator
from astropy.time import Time
from matplotlib.patches import FancyBboxPatch

from common import mycolor, myls, timelabel, plot_jbstyle


if __name__ == "__main__":
    parser = ap(description="Create ephemeris plots for multiple asteroids")
    parser.add_argument(
        "--outdir", type=str, default="fig",
        help="Output directory")
    parser.add_argument(
        "--f_ephem", nargs="+", required=True,
        help="Ephemeris files (one per object)")
    parser.add_argument(
        "--f_mpc", nargs="+", default=[],
        help="Sparse photometry files (one per object, optional)")
    parser.add_argument(
        "--ttype", type=str, default="week",
        help="Time in labels (hour, day, week, month)")
    parser.add_argument(
        "--outtype", default="pdf",
        help="Format of output figure")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    def extract_objname(path):
        base = os.path.basename(path)
        for key in ["2021TY14", "2021UW1", "2022GQ1"]:
            if key in base:
                return key
        return os.path.splitext(base)[0]

    range_dict = {
        "2021TY14": ("2021-10-10", 10),
        "2021UW1":  ("2021-10-24", 10),
        "2022GQ1":  ("2022-04-01", 10),
    }
    
    obsdate_dict = {
            "2021TY14": "2021-10-15 14:42:17",
            "2021UW1":  "2021-10-29 11:02:25",
            "2022GQ1":  "2022-04-07 11:04:31",
    }
    objtex_list = ["2021 TY$_{14}$", "2021 UW$_{1}$", "2022 GQ$_{1}$"]

    # Magnitude
    ylim_left = [21.5, 15]   
    # Phase angle
    ylim_right = [0, 180]  
    day_window = 6       

    plot_jbstyle()
    for i, ephem_file in enumerate(args.f_ephem):
        obj = extract_objname(ephem_file)
        objtex = objtex_list[i]
        print(f"  Processing {obj} ...")

        if not os.path.exists(ephem_file):
            print(f"  File not found: {ephem_file}")
            continue

        df = pd.read_csv(ephem_file, sep=",")

        # MPC 
        df_mpc = None
        if len(args.f_mpc) == len(args.f_ephem):
            mpc_file = args.f_mpc[i]
            if os.path.exists(mpc_file):
                df_mpc = pd.read_csv(mpc_file, sep=" ")
                print(f"  Loaded MPC file: {mpc_file}")

        t0_str, ndays = range_dict[obj]
        t0_jd = Time(t0_str, format="iso", scale="utc").jd
        t1_jd = t0_jd + ndays

        t_obs = obsdate_dict[obj]
        t_obs = datetime.datetime.strptime(obsdate_dict[obj], '%Y-%m-%d %H:%M:%S')
        
        df = df[(df["datetime_jd"] >= t0_jd) & (df["datetime_jd"] <= t1_jd)].reset_index(drop=True)

        fig = plt.figure(figsize=(6, 5))
        ax1 = fig.add_axes([0.08, 0.18, 0.85, 0.78])
        ax1 = timelabel(ax1, args.ttype)
        ax1.set_ylabel("Magnitude [mag]")
        # e.g. "2021" from "2021TY14"
        year_label = obj[:4] 
        ax1.set_xlabel(f"{year_label}")
        ax1.invert_yaxis()
        ax1.set_ylim(ylim_left)

        ax1.xaxis.set_major_locator(DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(DateFormatter("%m-%d"))

        ax1_r = ax1.twinx()
        ax1_r.set_ylabel("Phase angle [deg]", color=mycolor[4])
        ax1_r.set_ylim(ylim_right)
        ax1_r.spines['right'].set_color(mycolor[4])
        ax1_r.tick_params(axis='y', colors=mycolor[4])
        ax1.yaxis.set_major_locator(MultipleLocator(1.0))

        utclist = [datetime.datetime.strptime(i, '%Y-%b-%d %H:%M') for i in df["datetime_str"]]

        ax1.plot(utclist, df["V"], color="black", lw=2, ls=myls[0])
        ax1_r.plot(utclist, df["alpha"], color=mycolor[4], lw=2, ls=myls[1])

        ax1.vlines(t_obs, 0, 100, ls="dotted", color=mycolor[1], lw=2)

        if df_mpc is not None and len(df_mpc) > 0:
            df_mpc = df_mpc[df_mpc["rmsmag"] != 999].copy()
            df_mpc["obstime"] = pd.to_datetime(
                df_mpc["obstime"], format='ISO8601', errors='coerce', utc=True)
            df_mpc = df_mpc.dropna(subset=["obstime"])
            ax1.errorbar(
                df_mpc["obstime"], df_mpc["mag"], yerr=df_mpc["rmsmag"],
                color=mycolor[0], marker=" ", ls="None", zorder=100, lw=0.8)
            ax1.scatter(
                df_mpc["obstime"], df_mpc["mag"],
                color=mycolor[0], marker="o", facecolor="None",
                #label=f"MPC photometry of {obj}", zorder=101, lw=0.8)
                label=None, zorder=101, lw=0.8)

        # Put object name
        x_text, y_text = 0.25, 0.85
        ax1.text(
            x_text, y_text, objtex, color="black", size=24, zorder=1000,
            horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)
        # Add box 
        margin = 0.01
        box_width, box_height = 0.30, 0.10
        box_x = x_text - box_width / 2
        box_y = y_text - box_height / 2 + margin
        box = FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.02",
            linewidth=2,
            facecolor="white",
            edgecolor="black",
            transform=ax1.transAxes,
            zorder=999
        )
        ax1.add_patch(box)

        ax1.legend(fontsize=14, loc="upper left")

        out = os.path.join(args.outdir, f"{obj}_ephem.{args.outtype}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")
