#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot reduced magnitude vs. phase angle for multiple objects and bands.
"""
from argparse import ArgumentParser as ap
import os
import pandas as pd
import matplotlib.pyplot as plt

from common import mycolor, mymark, plot_jbstyle


def extract_objname(filename):
    base = os.path.basename(filename)
    return base.replace("MPCobs_", "").split("_")[0]


if __name__ == "__main__":
    parser = ap(description="Plot reduced magnitude vs. phase angle from MPC")
    parser.add_argument(
        "--f_mpc", nargs="+", required=True,
        help="MPC observation files (one per object)")
    parser.add_argument(
        "--outdir", type=str, default="fig",
        help="Output directory")
    parser.add_argument(
        "--outtype", type=str, default="pdf",
        help="Figure output format")
    args = parser.parse_args()
  
    os.makedirs(args.outdir, exist_ok=True)

    xylim_dict = {
        "2021TY14": {"x": (0, 60), "y": (30.0, 27.0)},
        "2021UW1":  {"x": (0, 110), "y": (30.0, 25.5)},
        "2022GQ1":  {"x": (0, 70), "y": (31.5, 27.8)},
    }

    # JPL H (absolute mag)
    H_dict = {
        "2021TY14": 27.27,
        "2021UW1": 26.16,
        "2022GQ1": 28.07,
    }
    
    plot_jbstyle()
    for f in args.f_mpc:
        df = pd.read_csv(f, sep=" ")
        # Remove wo/mag
        df = df[df["mag"]!=999]

        obj = extract_objname(f)
        df["obj"] = obj

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
        ax.set_xlabel("Phase angle [deg]")
        ax.set_ylabel("Reduced magnitude [mag]")
        ax.set_title(obj)

        bands = df["band"].unique()

        for idx_band, b in enumerate(bands):
            df_b = df[df["band"] == b]
            if len(df_b) == 0:
                continue
            label = f"{b}-band\nN={len(df_b)}"
            col = mycolor[idx_band]
            mark = mymark[idx_band]

            ax.scatter(
                df_b["alpha"], df_b["mag_red"],
                color=col, marker=mark, s=80, lw=1,
                facecolor="None", label=label)

        H = H_dict[obj]
        ax.scatter(
            0.1, H, marker="*", color=mycolor[0],
            edgecolor="black", s=200, zorder=1000, label=f"JPL H={H}")

        x0, x1 = xylim_dict[obj]["x"]
        y0, y1 = xylim_dict[obj]["y"]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)  

        ax.legend(fontsize=10)

        out_file = os.path.join(args.outdir, f"{obj}_pc.{args.outtype}")
        plt.savefig(out_file, dpi=200)
        plt.close()
        print(f"Saved: {out_file}")
