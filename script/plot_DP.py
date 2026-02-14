#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot Diameter--period relation of asteroids with spectral types.
"""
import os 
from argparse import ArgumentParser as ap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

from common import mycolor, plot_jbstyle, D_from_Hp
from hoya.core import extract_hoya, DBPATH

def show_ref_N(df):
    ref_counts = df["ref"].value_counts()
    print("Ref counts:")
    print(ref_counts)
    print()


def reduce_stype(df):
    """Reduce asteroid spectral type to S, C, X and others.

    Do not use multi type objects.

    Parameter
    ---------
    df: pandas.DataFrame
      should contain 'obj', 'stype'
    """
    types = df["stype"].unique()
    #print(f"Original types: {types}")
    # print(f"Original N={len(df)}")

    # Tholen
    # C : B, F, C, D, G, T
    # S : A, Q, R, S, V
    # X : E, M, P
    C_tholen = ["B", "F", "C", "D", "G", "T"]
    S_tholen = ["A", "Q", "R", "S", "V"]
    X_tholen = ["E", "M", "P"]

    # Bus (SMASS)
    # C : B,Cb,C,Cg,Ch,Cgh
    # S : A,K,L,Q,R, S,Sa,Sk,Sl,Sq,Sr
    # X : X,Xc,Xk,Xe
    # others : T, D, O, V, Ld
    C_bus = ["B", "Cb", "C", "Cg", "Ch", "Cgh"]
    S_bus = ["A", "K", "L", "Q", "R",
            "S", "Sa", "Sk", "Sl", "Sq", "Sr"]
    X_bus = ["X", "Xc", "Xk", "Xe"]
    other_bus = ["T", "D", "O", "V", "Ld"]

    # Bus-DeMeo
    # C : B,Cb,C,Cg,Ch,Cgh
    # S : A,K,L,Q,R, S(Sw), Sa, Sq(Sqw), Sr(Srw), Sv(Svw)
    # X : X,Xc,Xk,Xe
    # others : T, D, O, V(Vw)
    C_demeo = ["B", "Cb", "C", "Cg", "Ch", "Cgh"]
    S_demeo = ["A", "K", "Q", "R",
               "S", "Sw", "Sa", "Sq", "Sqw", "Sr", "Srw", "Sv", "Svw"]
    X_demeo = ["X", "Xc", "Xk", "Xe"]
    other_demeo = ["T", "D", "O", "V", "Vw", "L"]

    # Exception
    # Some objects have ambiguous or multiple classes
    C_exc = ["C,Ch", "C,F", "C_comp", "B"]
    S_exc = ["RS", "S,Sr", "Sq,Sr", "Srw", "S,Sr","Sr,Sq", "Sq,Q", "Qw",
             "S(IV)", "AS", "Q,R", "S_comp", "S;Sr", "Sx", "S;Sr", "Q,O", "S;Sr::", "Sq::", "S::", "Sq;Q", "R", "K", "S,V", "S;Sr"]
    X_exc = ["EM", "Xn", "Xe,Xk",]

    C_comp = C_tholen + C_bus + C_demeo + C_exc
    S_comp = S_tholen + S_bus + S_demeo + S_exc
    X_comp = X_tholen + X_bus + X_demeo + X_exc
    other = other_bus + other_demeo

    types_all = C_comp + S_comp + X_comp + other
    types_all =  list(set(types_all))

    # Do not use multi type objects
    # print("\n")
    union = set(types_all) | set(types)
    nouse = list(set(types_all) ^ union)
    print("  The stype below are not used.")
    print(nouse)
    # ignore  e.g.
    #    P,D  CX  C,X  D?  S,V
    #    C,P  TCG  -  D,X 0
    #    CQ  X,D D,X Q,O  U
    df = df[~df["stype"].isin(nouse)]
    
    # Save original
    df["stype_ori"] = df["stype"].copy()

    # Update types
    df["stype"] = df["stype"].replace(C_comp, "C")
    df["stype"] = df["stype"].replace(S_comp, "S")
    df["stype"] = df["stype"].replace(X_comp, "X")
    df["stype"] = df["stype"].replace(other, "other")
    return df


if __name__ == "__main__":
    parser = ap(description="Plot D-P relation.")
    parser.add_argument(
        "--indivi", action="store_true", default=False,
        help="Plot results from individual papers")
    parser.add_argument(
        "--Seimei", action="store_true", default=False,
        help="Plot results of Seimei photometry")
    parser.add_argument(
        "--outtype", type=str, default="pdf",
        help="format of output figures")
    parser.add_argument(
        "--outdir", type=str, default="fig",
        help="output directory")
    args = parser.parse_args()
    
    if not os.path.isdir(args.outdir):
      os.makedirs(args.outdir)


    # Setting for plot ========================================================
    plot_jbstyle()
    # Assumed albedo
    pv = 0.168
    pv_C = 0.07
    pv_S = 0.20
    pv_X = 0.10
    pv_E = 0.40

    df_neo = extract_hoya(DBPATH, "neoelems20220629")

    linestyles = {"S": "-", "C": "--", "X": ":"}
    offset = {"S":0.0, "C":0.15, "X":0.30}
    markers = {"S":"o", "C":"s", "X":"D", "O":"x"}
    colors  = {"S":mycolor[0], "C":mycolor[1], "X":mycolor[2]}
    fcolors  = {"S":"None", "C":mycolor[1], "X":mycolor[2]}
    
    # This slightly affects (Dmax_pca from 100 to 400?)
    Dmin_pca, Dmax_pca = 1, 200
    Pmin_pca, Pmax_pca = 1, 2*3600
    # Setting for plot ========================================================


    # 1. Rotation Period ======================================================
    # The latest LCDB
    df_LCDB = extract_hoya(DBPATH, "LCDB2023Oct")
    # TODO: Check Use only NEO !!
    neo_family = "9101"
    df_LCDB = df_LCDB[df_LCDB["family"]==neo_family]
    print(f"LCDB N={len(df_LCDB)} (NEO)")
    Umin = 2
    df_LCDB = df_LCDB[df_LCDB["U"].str[:1].astype(int) >= Umin]
    print(f"LCDB N={len(df_LCDB)} (U>={Umin})")

    # Remove stype in df_LCDB
    col_LCDB = df_LCDB.columns.tolist()
    col_use = [x  for x in col_LCDB if x!="stype"]
     #   ['obj', 'family', 'H', 'D', 'P', 'U', 'amin', 'amax', 'Pole', 'pv', 'stype']
    df_LCDB = df_LCDB[col_use]
    # Period in sec
    df_LCDB["P_sec"] = df_LCDB["P"]*3600.

    # Add diameter in m
    df_LCDB["D_m"] = 0.
    for idx,row in df_LCDB.iterrows():
        df_LCDB.at[idx, "D_m"] = D_from_Hp(row["H"], 0, pv, 0)[0]*1e3
    # 1. Rotation Period ======================================================


    # 2. Spectral types =======================================================
    # 2-1. Bus2002 
    # -> No objects are left! All updated after Bus2002.
    df_B02 = extract_hoya(DBPATH, "Bus2002")

    # First extract only NEA
    df_neo_2 = df_neo[df_neo["obj"].isin(df_B02["obj"])]
    obj_B02 = set(df_B02.obj.tolist())
    obj_neo = set(df_neo_2.obj.tolist())
    obj_common = obj_B02 & obj_neo
    df_B02 = df_B02[df_B02["obj"].isin(obj_common)]
    
    # Sort to add H properly
    df_B02 = df_B02.sort_values("obj")
    df_B02 = df_B02.reset_index(drop=True)
    df_neo_2 = df_neo_2.sort_values("obj")
    df_neo_2 = df_neo_2.reset_index(drop=True)
    assert len(df_B02) == len(df_neo_2), "Check the code."
    df_B02["H"] = df_neo_2["H"]

    # 2-2. Binzel2004
    df_B04 = extract_hoya(DBPATH, "Binzel2004")

    # 2-3. Devogele2019
    df_D19 = extract_hoya(DBPATH, "Devogele2019")
    # Remove duplicated
    df_D19 = df_D19[~df_D19.duplicated(subset="obj")]
    # Add H
    df_neo_1 = df_neo[df_neo["obj"].isin(df_D19["obj"])]
    obj_D19 = set(df_D19.obj.tolist())
    obj_neo = set(df_neo_1.obj.tolist())
    obj_common = obj_D19 & obj_neo
    df_D19_only = df_D19[~df_D19["obj"].isin(obj_common)]
    df_D19 = df_D19.sort_values("obj")
    df_neo_1 = df_neo_1.sort_values("obj")
    df_D19 = df_D19.reset_index(drop=True)
    df_neo_1 = df_neo_1.reset_index(drop=True)
    df_D19["H"] = df_neo_1["H"]

    # 2-4. Binzel2019
    df_B19 = extract_hoya(DBPATH, "Binzel2019")

    # 2-5. Marsset2022
    df_M22 = extract_hoya(DBPATH, "Marsset2022")


    # 2-6. Mommert2016
    df_M16 = extract_hoya(DBPATH, "Mommert2016")

    # 2-7. Sergeyev2021
    df_S21 = extract_hoya(DBPATH, "Sergeyev2021")
    # Select high possibility objects
    p_th = 0.8
    df_S21 = df_S21[df_S21["p_stype"].astype(float) > p_th]
    # We don't use U
    stype_S21_notuse = ["U"]
    df_S21 = df_S21[~df_S21["stype"].isin(stype_S21_notuse)]
    
    # 2-8. Sanchez2024
    df_S24 = extract_hoya(DBPATH, "Sanchez2024")
    # Add H
    df_neo_2 = df_neo[df_neo["obj"].isin(df_S24["obj"])]
    obj_S24 = set(df_S24.obj.tolist())
    obj_neo = set(df_neo_2.obj.tolist())
    obj_common = obj_S24 & obj_neo
    df_S24_only = df_S24[~df_S24["obj"].isin(obj_common)]
    df_S24 = df_S24.sort_values("obj")
    df_neo_2 = df_neo_2.sort_values("obj")
    df_S24 = df_S24.reset_index(drop=True)
    df_neo_2 = df_neo_2.reset_index(drop=True)
    df_S24["H"] = df_neo_2["H"]


    # Add references
    df_B02["ref"] = "Bus2002"
    df_B04["ref"] = "Binzel2004"
    df_B19["ref"] = "Binzel2019"
    df_D19["ref"] = "Devogele2019"
    df_M22["ref"] = "Marsset2022"
    df_M16["ref"] = "Mommert2016"
    df_S21["ref"] = "Sergeyev2021"
    df_S24["ref"] = "Sanchez2024"
    
    # Order is important.
    # We prioritize newer data, which should be at the end
    #   Devogele+2019: published 2019 October 23
    #   Bolin+2019:    Available online 18 December 2018
    dfs_spec = [df_B02, df_B04, df_M16, df_B19, df_D19, df_S21, df_M22, df_S24]

    # Merge spec
    col_use = ["obj", "stype", "H", "ref"]
    for i in range(len(dfs_spec)):
        dfs_spec[i] = dfs_spec[i][col_use]
    
    # Concatenate all DataFrames
    df_spec = pd.concat(dfs_spec, axis=0, ignore_index=True)
    
    # Count per reference before removing duplicates
    show_ref_N(df_spec)
    
    # Remove duplicated objects, keeping the last occurrence (latest data)
    df_spec = df_spec[~df_spec.duplicated(subset="obj", keep="last")]
    
    # Count per reference after removing duplicates
    ref_counts = df_spec["ref"].value_counts()
    print("After removing duplicates (latest kept)")
    show_ref_N(df_spec)
    
    
    types0 = df_spec["stype"].unique()
    print(f"Original stypes: {types0}")
    print()

    df_spec = reduce_stype(df_spec)
    types1 = df_spec["stype"].unique()
    # Some ambiguous data is removed here (e.g., stype == 0)
    print(f"Reduced spectral types: {types1}")
    print()

    print("After reducing stypes")
    show_ref_N(df_spec)
    # 2. Spectral types =======================================================


    # Merge 1 (rotP) and 2 (Spec)
    print(f"  Merge df_LCDB and df_spec")
    df_concat = pd.merge(df_spec, df_LCDB, on="obj")
    # Use H in LCDB
    df_concat["H"] = df_concat["H_y"]
    show_ref_N(df_concat)


    df_C = df_concat[df_concat["stype"]=="C"]
    df_S = df_concat[df_concat["stype"]=="S"]
    df_X = df_concat[df_concat["stype"]=="X"]
    df_other = df_concat[
        (df_concat["stype"]!="S") &
        (df_concat["stype"]!="C") &
        (df_concat["stype"]!="X")
    ]

    # Re-Calculate diameters depending on albedo
    df_C["D_m"] = [D_from_Hp(H, 0, pv_C, 0)[0]*1e3 for H in df_C["H"]]
    df_S["D_m"] = [D_from_Hp(H, 0, pv_S, 0)[0]*1e3 for H in df_S["H"]]
    df_X["D_m"] = [D_from_Hp(H, 0, pv_X, 0)[0]*1e3 for H in df_X["H"]]
    df_other["D_m"] = [D_from_Hp(H, 0, pv, 0)[0]*1e3 for H in df_other["H"]]

    print(f"-> df_concat (N={len(df_concat)})")
    print(f"     S-type   : N={len(df_S)}")
    print(f"     C-type   : N={len(df_C)}")
    print(f"     X-type   : N={len(df_X)}")
    print(f"     others   : N={len(df_other)}")
    print()


    # Indivisual papers (by hand)
    # TODO: Check these are not included in df_concat
    # Maybe U flag is less than 2
    if args.indivi:
        # Some results are missing in LCDB2023Oct.
        # Collect individual sources with periods less than 1 hr
        # Note: 
        # Already included
        # 1. Reddy2016 
        #    2015 TC25, E-type, P=133, H=29.5
        # 2. Jenniskens2009
        #    2008 TC3, C-type, 97 s

        df_indivi_list = []
        # Polishook2012 ========================================================
        # This is not included in df_concat,
        # while 2012 KT42 is included.
        # D = 20
        obj = ["2012KP24"]
        H   = [26.4]
        stype = ["C"]
        P_sec = [2.5008 * 60]  
        D_m = [20] 
        P12 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Polishook2012"
        ))
        df_indivi_list.append(P12)
        # Polishook2012 ========================================================
        
        # Kwiatkowski2021 =====================================================
        # This is not included in df_concat
        # D = 30
        obj = ["2021DW1"]
        H   = [24.8]
        stype = ["S"]
        P_sec = [0.013760 * 3600]  
        D_m = [30] 
        K21 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Kwiatkowski2021"
        ))
        df_indivi_list.append(K21)
        # Kwiatkowski2021 =====================================================


        # Fenucci2021 =========================================================
        # These are not included in df_concat
        # P ~ 11 min
        obj = ["2011PT"]
        H   = [23.9]
        stype = ["X"]
        P_sec = [0.17 * 3600]
        D_m = [35]
        F21 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Fenucci2021"
        ))
        df_indivi_list.append(F21)
        # Fenucci2021 =========================================================


        # Licandro2023 ========================================================
        # This is not included in df_concat
        # We do not include 2022 AB since spectrum is unusual
        obj = ["2021NY1"]
        H   = [21.84]
        stype = ["S"]
        P_sec = [13.3449*60]
        # They said "smaller than 120 m"
        D_m = [0]
        L23 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Licandro2023"
        ))
        df_indivi_list.append(L23)
        # Licandro2023 ========================================================


        # Bolin2024 ===========================================================
        # These are not included in df_concat
        # also Fenucci+2023: 2016 GE1
        obj = ["2016GE1", "2016CG18", "2016EV84"]
        H   = [28.5, 26.7, 26.7]
        stype = ["S", "X", "X"]
        P_sec = [55.180, 52.237, 30.664]
        # No estimate in the paper
        D_m = [0, 0, 0]
        B24 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Bolin2024"
        ))
        df_indivi_list.append(B24)
        # Bolin2024 ===========================================================


        # Zambrano-Marin2022 ==================================================
        # This is not included in df_concat
        # Both period and spectral type from radar!
        # Ah, but we cannot break the degenercy between S and C.
        #obj = ["2019OK"]
        #H   = [23.3]
        ## between 70 to 130 m
        #D_m = [100]
        #stype = ["CorS"]
        #p_sec = [0.00666 * 3600]
        #z22 = pd.DataFrame(dict(
        #    obj=obj, H=H, stype=stype, P_sec=P_sec, ref="Zambrano-Marin2022"
        #))
        #df_indivi_list.append(Z22)
        # Zambrano-Marin2024 =================================================


        # Zambrano-Marin2024 =================================================
        # Based on the Fig. 5 of Zambrano-Marin2024.
        # Add two E-types as X-types
        obj = ["1999TY2", "2000UK11", "2006AM4", "2013QR1", "2013XY8",
               "2014TV", "2014WU200", "2015AK45", "2015HS11", "2015RF36",
               "2015SZ2", "2015XA379", "2016GS2", "2016RF36", "2015SZ2",
               "2015XA379", "2016GS2", "2016RD34", "2017EK", "2017KJ27",
               "2017LE", "2018LK", "2019NN3", "2020KB3", "2001OE84", 
               "2000GD65"]
        P_min = [7.28, 1.60, 5.08, 2.82, 3.64, 
                 1.31, 1.07, 1.55, 1.16, 0.74,
                 2.30, 2.31, 0.76, 1.38, 0.38,
                 2.16, 1.69, 7.38, 2.25, 3.77, 
                 29.16, 117.17] 

        obj = ["2017EK", "2018LK"]
        H = [24.2, 21.8]
        # Adopted by them as well
        D_m = [30, 0]
        P_min = [0.38, 7.38]
        # Both E-type!
        stype = ["X", "X"]
        P_sec = [x*60 for x in P_min]
        Z24 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Zambrano-Marin2024"
        ))
        df_indivi_list.append(Z24)
        # Zambrano-Marin2024 =================================================


        # Santana-Ros2025 =====================================================
        # These are not included in df_concat
        # D=11 m
        obj = ["1998KY26"]
        H   = [26.2]
        D_m = [11]
        # E-type
        stype = ["X"]
        P_sec = [5.3516*60]
        S25 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec, D_m=D_m, ref="Santana-Ros2025"
        ))
        df_indivi_list.append(S25)
        # Santana-Ros2025 =====================================================


        # Greenstreet2026 =====================================================
        # These are not included in df_concat
        # N = 9 with P < 1 hr in Table 2
        # - Do not include samples with different rotP in two methods (e.g. MJ79)
        # - Only 1, 2025MJ71, is used in the analysis below. 
        #   Others are larger than 200 m
        obj = ["2025ME68", "2025MG56", "2025MJ71", "2025MK41", "2025MN25",
               "2025MN45", "2025MU15", "2025MU8", "2025MV71"]
        H   = [21.2, 18.9, 22.7, 19.3, 20.5, 
               18.7, 21.6, 19.0, 18.2]
        P_hr = [0.9, 0.3, 0.0031, 0.063, 0.4,
                0.031, 0.4, 0.8, 0.2]

        P_sec = [x*3660 for x in P_hr]
        # By visual inspection 
        stype = ["X", "C", "X", "C", "C",
                  "C", "X", "C", "C"]
        G26 = pd.DataFrame(dict(
            obj=obj, H=H, stype=stype, P_sec=P_sec,  ref="Greenstreet2026"
        ))
        G26["D_m"] = 0
        # Calculate diameter
        df_indivi_list.append(G26)
        # Greenstreet2026 =====================================================


        # Merge all
        df_indivi = pd.concat(df_indivi_list, axis=0, ignore_index=True)
        df_indivi["stype_ori"] = df_indivi["stype"]
        print(f"-> df_indivi (N={len(df_indivi)})")
        print(df_indivi)
        print()

        # Compute diameters depending on albedo
        for idx,row in df_indivi.iterrows():
            # Already derived
            if row["D_m"] != 0:
                continue
            stype = row["stype"]
            if stype == "S":
                p = pv_S
            elif stype == "C":
                p = pv_C
            elif stype == "X":
                p = pv_X
            df_indivi.at[idx, "D_m"] = D_from_Hp(row["H"], 0, p, 0)[0]*1e3

        df_indivi_C     = df_indivi[df_indivi["stype"]=="C"]
        df_indivi_S     = df_indivi[df_indivi["stype"]=="S"]
        df_indivi_X = df_indivi[df_indivi["stype"]=="X"]
        df_indivi_other = df_indivi[
            (df_indivi["stype"]!="S") &
            (df_indivi["stype"]!="C") &
            (df_indivi["stype"]!="X")
        ]


    # Add 2021 TY14, 2021 UW1, and 2022 GQ1
    # Input has obj, H, stype, P_sec
    if args.Seimei:
        obj_list    = ["2021TY14", "2021UW1", "2022GQ1"]
        H_list      = [27.27, 28.16, 28.07]
        P_list      =  [15.292, 0.005862*3600, 8.774] 
        stype_list  =  ["C", "S", "S"] 
        df_Seimei = pd.DataFrame(dict(obj=obj_list, H=H_list, P_sec=P_list, stype=stype_list))
        df_Seimei["stype_ori"] = df_Seimei["stype"]

        # Compute diameters depending on albedo
        for idx,row in df_Seimei.iterrows():
            stype = row["stype"]
            if stype == "S":
                p = pv_S
            elif stype == "C":
                p = pv_C
            elif stype == "X":
                p = pv_X
            df_Seimei.at[idx, "D_m"] = D_from_Hp(row["H"], 0, p, 0)[0]*1e3

        df_Seimei["ref"] = "Seimei"

        df_TY = df_Seimei[df_Seimei["obj"] == "2021TY14"]
        df_UW = df_Seimei[df_Seimei["obj"] == "2021UW1"]
        df_GQ = df_Seimei[df_Seimei["obj"] == "2022GQ1"]

        print(f"-> df_Seimei (N={len(df_Seimei)})")
        print(df_Seimei)
        print()


    # Merge df_concat, df_indivi, df_S
    all_dfs = [df_concat]
    #assert False, df_concat[df_concat["obj"] == "2012KP24"]
    if args.indivi:
        all_dfs.append(df_indivi)
    if args.Seimei:
        all_dfs.append(df_Seimei)
    
    # Merge all
    df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # Remove duplicates by obj
    df_all = df_all.drop_duplicates(subset="obj")
    
    # Keep only selected columns
    df_all = df_all[["obj", "stype", "P_sec", "H", "D_m", "ref", "stype_ori"]]
    
    # Optional: sort by diameter
    df_all = df_all.sort_values("D_m")
    df_all["D_m"] = df_all["D_m"].round(1)
    df_all["P_sec"] = df_all["P_sec"].round(1)

    # Save to space-delimited file
    # TODO: Update diameter. Current D does not consider albedo
    outfile = "../data/ufra_DP_merged.txt"
    df_all.to_csv(outfile, index=False, sep=" ")
    print(f"Merged data saved: {outfile}")
    print()

    # Output notable objects
    ## Faster than 60 sec
    ## df_Seimei N=3
    ## df_indivi N=5
    ## df_concat N=5 as of Dec. 9, 2025
    ##   2021CG   P=15.3s, S (LCDB = Beniyama+2022, IRTF spec)
    ##   2014TP57 P=49.3s, C (LCDB, MANOS spec)
    ##   2017FJ   P=59.4s, S (LCDB, MANOS spec)
    ##   2017FK   P=15.4s, S (LCDB, MANOS spec)
    ##   2017QG18 P=11.9s, S (LCDB=MANOS, MANOS spec)
    df_fast = df_concat[df_concat["P_sec"] < 60]
    print("Faster than 60 sec in df_concat")
    print(df_fast)

    
    # Plot 1. D-P diagram =====================================================
    fig = plt.figure(figsize=(16, 6))
    # 1. Plot LCDB Object w/ and wo/ spectral info.
    ax1 = fig.add_axes([0.10, 0.15, 0.35, 0.8])
    # 2 D-P with spectral type (S vs C)
    ax2 = fig.add_axes([0.6, 0.15, 0.35, 0.8])

    ax1.text(-0.20, 0.98, "(a)", size=22, transform=ax1.transAxes)
    ax2.text(-0.20, 0.98, "(b)", size=22, transform=ax2.transAxes)


    ax1.scatter(
        df_LCDB["D_m"], df_LCDB["P_sec"], s=20, marker="+", color="black",
        label=f"LCDB NEAs N={len(df_LCDB)}")

    # S-type
    ax2.scatter(
        df_S["D_m"], df_S["P_sec"],  
        s=100, marker=markers["S"], linewidth=1,
        edgecolor=colors["S"], facecolor=fcolors["S"], label=f"S-type N={len(df_S)}")
    # C-type
    ax2.scatter(
        df_C["D_m"], df_C["P_sec"], 
        s=60, marker=markers["C"], lw=1,
        edgecolor=colors["C"],  facecolor=fcolors["C"], label=f"C-type N={len(df_C)}")
    # X-type
    ax2.scatter(
        df_X["D_m"], df_X["P_sec"],
        s=60, marker=markers["X"], linewidth=1,
        edgecolor=colors["X"], facecolor=fcolors["X"],
        label=f"X-type N={len(df_X)}"
    )
    # Others
    #   Ld-types: Camillo, 1999TX16,
    #   O-types : 2000BM19, 2014JD
    assert len(df_other) == 4, "Check"
    ax2.scatter(
        df_other["D_m"], df_other["P_sec"], marker=markers["O"], s=20, linewidth=1,
        zorder=-1,facecolor="black", label=f"Others")

    # Can be merged to df_S, df_C, and df_X 
    if args.indivi:
        ax2.scatter(
            df_indivi_S["D_m"], df_indivi_S["P_sec"],
            marker=markers["S"], s=100, linewidth=1,
            zorder=-1, ec=colors["S"], facecolor="None",
        )
        ax2.scatter(
            df_indivi_C["D_m"], df_indivi_C["P_sec"],
            marker=markers["C"], s=100, linewidth=1,
            ec=colors["C"], facecolor="None",
        )
        ax2.scatter(
            df_indivi_X["D_m"], df_indivi_X["P_sec"],
            marker=markers["X"], s=100, linewidth=1,
            ec=colors["X"], facecolor="None",
        )
        assert len(df_indivi_other) == 0, "Check"
     

    if args.Seimei:
        for ax in [ax1, ax2]:
            # S-types
            ax.scatter(
              df_UW["D_m"], df_UW["P_sec"], marker="p", s=200, facecolor=colors["S"],
              edgecolor="black", label=r"2021 UW$_{1}$", zorder=1)
            ax.scatter(
              df_GQ["D_m"], df_GQ["P_sec"], marker="h", s=200, facecolor=colors["S"],
              edgecolor="black", label=r"2022 GQ$_{1}$", zorder=1)
            # C-types
            ax.scatter(
              df_TY["D_m"], df_TY["P_sec"], marker="8", s=200, facecolor=colors["C"],
              edgecolor="black", label=r"2021 TY$_{14}$", zorder=1)


    for ax in [ax1, ax2]:
        ax.set_xlabel("Diameter [m]")
        ax.set_ylabel("Rotation period [s]")
        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_yticks([1e4, 1e3, 1e2, 1e1, 1e0])
        ax.set_yticklabels(["$10^{4}$", "$10^{3}$", "$10^{2}$", "$10^{1}$", "1"])

        ax.set_xticks([1, 10, 100, 1000, 1e4, 1e5, 1e6])
        ax.set_xticklabels(["1", "10", "$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$", "$10^{6}$"])
        Dmin, Dmax = 1, 1e4
        Pmin, Pmax = 1, 36000
        ax.set_xlim([Dmin, Dmax])
        ax.set_ylim([Pmax, Pmin])
        ax.legend(loc="upper right")

    
    out = os.path.join(args.outdir, f"DP.{args.outtype}")
    fig.savefig(out)
    plt.close()
    # Plot 1. D-P diagram =====================================================
