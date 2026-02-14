#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle Seimei observations for multiple objects.

!!! This script is for JB's MacBook. !!!

Data:
1. 2021 TY14
   2021-10-15, gri, 1s, N=875 frames, rad=15 pix
2. 2021 UW1
   (2021-10-28, gri, not used)
   2021-10-29, gri, 5s, N=553 frames, rad=15 pix
3. 2022 GQ1 (not used here)
   This is tricky. 
   2022-04-07, gri, 1s, N=80 frames
     Colors: 20 s stacked, N=3 (out of 4) used, rad=10, see jupyter notebook
     Lightcurves: 1 s, N=80, rad=10

Outputs:
    gri_col.txt  -> "jd", "g_red", "gerr", "r_red", "rerr", "alpha"
                    (already 'ltcor')
    gri_mag.txt -> "jd", "g_red", "gerr", "r_red", "rerr",
                    "i_red", "ierr", "alpha"
                    (already 'ltcor' and 'distance cor')
"""
import os
from argparse import ArgumentParser as ap
import pandas as pd


def load_photometry_files(file_list, mode):
    """
    Load and concatenate photometry data files for given mode ('color' or 'mag').

    Parameters
    ----------
    file_list : list[str]
        List of input text file paths.
    mode : str
        Either 'color' or 'mag'.

    Returns
    -------
    df_concat : pd.DataFrame
        Concatenated dataframe of all observations.
    Nimg_list : list[int]
        List of image counts per observation.
    """
    df_list = []
    Nimg_list = []

    for idx_csv, path in enumerate(file_list):
        df = pd.read_csv(path, sep=" ", index_col=(0 if mode == "color" else None))
        df["n_obs"] = idx_csv + 1
        df = df.reset_index(drop=True)

        if idx_csv == 0:
            col_stan = df.columns.tolist()
        else:
            df = df.reindex(columns=col_stan)

        df_list.append(df)
        Nimg_list.append(len(df))

    df_concat = pd.concat(df_list)
    return df_concat, Nimg_list


def summarize_obs(df, label):
    """Print summary of observation data."""
    N = len(df)
    print(f"  {label}")
    print(f"    N_all = {N}")
    for x in sorted(set(df.n_obs)):
        df_temp = df[df["n_obs"] == x]
        t0 = df_temp["t_utc_g"].iloc[0]
        t1 = df_temp["t_utc_g"].iloc[-1]
        N_sub = len(df_temp)
        print(f"    N_{x}, t0, t1 = {N_sub}, {t0}, {t1}")


def process_target_col(target_name, paths_col):
    # ----- Color -----
    df_c, Nimg_col_list = load_photometry_files(paths_col, mode="color")
    summarize_obs(df_c, "Color")
    df_c["obj"] = target_name
    return df_c


def process_target(target_name, paths_col, paths_mag):
    """Process one target (color and magnitude) and return DataFrames."""
    print(f"\nProcessing target: {target_name}")

    # ----- Color -----
    df_c, Nimg_col_list = load_photometry_files(paths_col, mode="color")
    summarize_obs(df_c, "Color")
    df_c["obj"] = target_name

    # ----- Magnitude -----
    df_m, Nimg_mag_list = load_photometry_files(paths_mag, mode="mag")
    summarize_obs(df_m, "Magnitude")
    df_m["obj"] = target_name

    # ----- Consistency check -----
    assert Nimg_mag_list == Nimg_col_list, f"[{target_name}] Check the input"
    print(f"    Effective images N = {Nimg_mag_list}")

    return df_c, df_m


if __name__ == "__main__":
    parser = ap(description="Handle Seimei data of multiple objects.")
    parser.add_argument(
        "--outdir", type=str, default="data",
        help="Directory for output")
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Define targets =========================================================
    targets = {
        "2021TY14": {
            "color": [
                "/Users/beniyama/research/tricolor_three_nea/phot/2021TY14_20211015/2021TY14_N875_rad15_20251201/plotcolor/photres_obj_magerr0f05dedge100eflag1_photres_ref_magerr0f05dedge100eflag1_Nobjmin3Nframemin50_mag0f0_0f0_0f0to17f0_17f0_17f0_snrmin10/colall_N867_eff814_photres_obj_magerr0f05dedge100eflag1.txt",
            ],
            "mag": [
                "/Users/beniyama/research/tricolor_three_nea/phot/2021TY14_20211015/2021TY14_N875_rad15_20251201/plotmag/photres_obj_magerr0f05dedge100eflag1_photres_ref_magerr0f05dedge100eflag1_Nobjmin3Nframemin50_mag0_0_0to20_20_20_snrmin10/magall_N867_photres_obj_magerr0f05dedge100eflag1.txt",
            ]
        },
        "2021UW1": {
            "color": [
                "/Users/beniyama/research/tricolor_three_nea/phot/2021UW1_20211029/2021UW1_N553_20251114_rad15/plotcolor/photres_obj_magerr0f05dedge100eflag1_photres_ref_magerr0f05dedge100eflag1_Nobjmin3Nframemin10_mag0_0_0to20_20_20_snrmin10/colall_N459_eff453_photres_obj_magerr0f05dedge100eflag1.txt",
            ],
            "mag": [
                "/Users/beniyama/research/tricolor_three_nea/phot/2021UW1_20211029/2021UW1_N553_20251114_rad15/plotmag/photres_obj_magerr0f05dedge100eflag1_photres_ref_magerr0f05dedge100eflag1_Nobjmin3Nframemin10_mag0_0_0to20_20_20_snrmin10/magall_N459_photres_obj_magerr0f05dedge100eflag1.txt",
            ]
        },
    }

    df_all_col = []
    df_all_mag = []

    for target, paths in targets.items():
        if paths["mag"] == []:
            # Only color
            df_c = process_target_col(target, paths["color"])
            df_all_col.append(df_c)
        else:
            df_c, df_m = process_target(target, paths["color"], paths["mag"])
            df_all_col.append(df_c)
            df_all_mag.append(df_m)

    # Merge all targets ======================================================
    df_col_all = pd.concat(df_all_col).reset_index(drop=True)
    df_mag_all = pd.concat(df_all_mag).reset_index(drop=True)

    # Save merged outputs ====================================================
    out_col = os.path.join(args.outdir, "gri_col.txt")
    out_mag = os.path.join(args.outdir, "gri_mag.txt")
    df_col_all.to_csv(out_col, sep=" ", index=False)
    df_mag_all.to_csv(out_mag, sep=" ", index=False)

    print("\n  All objects processed and merged successfully.")
    print(f"  -> {out_col}")
    print(f"  -> {out_mag}")
