#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check magerr (SNR) of lightcurves.
"""
from argparse import ArgumentParser as ap
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = ap(description="Check SNR of lightcurves.")
    parser.add_argument(
        "res", type=str, 
        help="Photometric results (gri_mag.txt)")
    parser.add_argument(
        "res_GQ", type=str, 
        help="Photometric results (gri_2022GQ1_1s_N80.txt)")
    args = parser.parse_args()


    # Read combined data
    df = pd.read_csv(args.res, sep=" ")
    df_GQ = pd.read_csv(args.res_GQ, sep=" ")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    targets = [
        ("2021TY14", df),
        ("2021UW1", df),
        ("2022GQ1",  df_GQ),
    ]
    
    keys_mag = ["gmag", "rmag", "imag"]
    keys_magerr = ["gmagerr", "rmagerr", "imagerr"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    for idx, (obj, df_use) in enumerate(targets):
        ax = axes[idx]
        print(obj)
    
        df_obj = df_use[df_use["obj"] == obj]
    
        for keymag, keymagerr, c in zip(keys_mag, keys_magerr, colors):
            magerr = df_obj[keymagerr].dropna()
    
            ax.hist(
                magerr,
                bins=30,
                histtype="step",
                linewidth=1.5,
                label=keymag.replace("mag", ""),
                color=c,
            )
    
        ax.set_title(obj)
        ax.set_xlabel("mag error")
        ax.set_yscale("log")
        ax.legend()
    
    axes[0].set_ylabel("N")
    
    plt.tight_layout()
    plt.show()

