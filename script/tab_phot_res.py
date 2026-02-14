#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Observations
Create tables about observing condition and results.
The result of query to JPL is saved just in case.
Check obsdate0 and obsdate1 before submittion.
Output physical properties using JPL data as well.
"""
import os
from argparse import ArgumentParser as ap
import datetime
import pandas as pd
import numpy as np

from common import (
    loc_Seimei, add_obsinfo, tab_phot, tab_res, color_nea, rotation_nea)


if __name__ == "__main__":
    parser = ap(description='Obtain info. and create table.')
    parser.add_argument(
        "res", type=str, 
        help="Photometric results (color or mag)")
    parser.add_argument(
        "res_GQ", type=str, 
        help="Photometric results of 2022 GQ1 (not stacked)")
    parser.add_argument(
        "--date", type=str, default=None,
        help="date to reuse the aspect data")
    parser.add_argument(
        "--outdir", type=str, default="tab",
        help="output directory")
    args = parser.parse_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    today = datetime.date.today()
    
    # For 2021UW1 and 2021TY14
    df = pd.read_csv(args.res, sep=" ")
    # For 2022GQ1
    df_GQ = pd.read_csv(args.res_GQ, sep=" ")

    obj_list = ["2021 TY14", "2021 UW1", "2022 GQ1"]
    objtex_list = ["2021 TY$_{14}$", "2021 UW$_1$", "2022 GQ$_1$"]
    texp_list = [1, 5, 1]
    # 2021 TY14: 9  pix ~ 3.15 ~ 3.2 arcsec 
    # 2021 UW1 : 10 pix ~ 3.5  ~ 3.5 arcsec
    # 2022 GQ1 : 7  pix ~ 2.45 ~ 2.5 arcsec
    seeing_list = [3.2, 3.5, 2.5]

    if not args.date:
        # Colors
        # 1. 2021 TY14
        # 2. 2021 UW1
        # 3. 2022 GQ1
        
        df_list = []

        # 2021UW1 and 2021TY14
        for idx_obj, obj in enumerate(["2021TY14", "2021UW1"]):
            df_obj = df[(df["obj"] == obj)]
            df_obj = df_obj.reset_index(drop=True)
            # Use r-band time as standard 

            # Number of effetive images
            Nimg = len(df_obj)

            # Starting and ending time
            # in 'midtime of exposure' 
            idx0, idx1 = 0, Nimg - 1
            # Note: r-band timestamps in some data have been replaced with those of r-band
            #       due to problems in timestamps.
            t0 = df_obj.at[idx0, "t_utc_r"]
            t1 = df_obj.at[idx1, "t_utc_r"]
            print(f"t0, t1 = {t0}, {t1}")

            # Create dataframe
            df_obj = pd.DataFrame(dict(
                obsdate0=[t0], obsdate1=[t1], 
                Nimg=[Nimg], t_exp=texp_list[idx_obj], seeing=seeing_list[idx_obj]
                ))

            # JPL style
            df_obj["obj"] = obj_list[idx_obj]
            df_obj["objtex"] = objtex_list[idx_obj]
            
            # Add obs info.
            df_obj = add_obsinfo(df_obj, loc_Seimei)
            df_obj = df_obj.reset_index(drop=True)
            df_list.append(df_obj)


        # 2022GQ1
        df_obj = df_GQ.reset_index(drop=True)
        # One radius
        rad_min = np.min(df_GQ["radius"])
        df_GQ = df_GQ[df_GQ["radius"] == rad_min]
        df_obj = df_GQ.reset_index(drop=True)

        # Number of effetive images
        Nimg = len(df_obj)

        # Starting and ending time
        # in 'midtime of exposure' 
        idx0, idx1 = 0, Nimg - 1
        # Note: r-band timestamps in some data have been replaced with those of r-band
        #       due to problems in timestamps.

        t0 = df_obj.at[idx0, "t_utc_r"]
        t1 = df_obj.at[idx1, "t_utc_r"]
        print(f"t0, t1 = {t0}, {t1}")

        # Create dataframe
        df_obj = pd.DataFrame(dict(
            obsdate0=[t0], obsdate1=[t1], 
            Nimg=[Nimg], t_exp=texp_list[2], seeing=seeing_list[2]
            ))

        # JPL style
        df_obj["obj"] = obj_list[2]
        df_obj["objtex"] = objtex_list[2]
        
        # Add obs info.
        df_obj = add_obsinfo(df_obj, loc_Seimei)
        df_obj = df_obj.reset_index(drop=True)
        df_list.append(df_obj)

        
        df = pd.concat(df_list)

        # Save 19-columns info. for logging
        out = f"info_with_JPL_{today}.txt"
        out = os.path.join(args.outdir, out)
        df.to_csv(out, sep=" ", index=False)
        date = str(today)
    else:
        print(f"Reuse aspect data obtained on {args.date}.")
        f = f"info_with_JPL_{args.date}.txt"
        f = os.path.join(args.outdir, f)
        df = pd.read_csv(f, sep=" ")
        date = args.date

    df = df.reset_index(drop=True)

    # create obs tex table
    out = f"tab_phot.tex"
    out = os.path.join(args.outdir, out)
    tab_phot(df, date, out)
    
    # Add colors and rotP
    for idx, row in df.iterrows():
        obj = row["obj"].replace(" ", "")
        gr, grerr, ri, rierr = color_nea(obj)
        df.loc[idx, "g_r"] = gr
        df.loc[idx, "g_rerr"] = grerr
        df.loc[idx, "r_i"] = ri
        df.loc[idx, "r_ierr"] = rierr

        P, Perr, dm, dmerr = rotation_nea(obj)
        df.loc[idx, "P"] = P
        df.loc[idx, "Perr"] = Perr
        df.loc[idx, "dm"] = dm
        df.loc[idx, "dmerr"] = dmerr
    out = f"tab_res.tex"
    out = os.path.join(args.outdir, out)
    tab_res(df, date, out)
