# Simultaneous Tricolor Video Observations of Three Tiny Near-Earth Asteroids with Sub-Minute Rotation Periods
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[developer mail](mailto:jbeniyama@oca.eu)

## Overview
This is a repository for a paper in prep.
Figures are made in `./plot`.


## Structure 
```
./
  README.md
  data/
  fig/
  script/
  tab/
  tex/
  .gitignored
```

## Data (in /data)
* `2021TY14_2013-01-01to2023-01-01step1hcode500eph.txt` (ephemeris of 2021 TY14)
* `2021UW1_2013-01-01to2023-01-01step1hcode500eph.txt` (ephemeris of 2021 UW1)
* `2022GQ1_2013-01-01to2023-01-01step1hcode500eph.txt` (ephemeris of 2022 GQ1)
* `MPCobs_2021TY14_20251031.txt` (observations of 2021 TY14 in MPC)
* `MPCobs_2021UW1_20251031.txt` (observations of 2021 UW1 in MPC)
* `MPCobs_2022GQ1_20251031.txt` (observations of 2022 GQ1 in MPC)
* `gri_col.txt` (colors of NEAs obtained with Seimei)
* `gri_mag.txt` (absolute magnitudes of NEAs obtained with Seimei)
* `gri_2022GQ1_1s_N80.txt` (relative lightcurve of 2022GQ1)


## Preprocesses
```
# Make `requirements.txt` with `pipreqs . --force`, and change the versions in it.
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python script/setup.py
# Get photometry of 2022 GQ1 on `jupyter notebook`.
```


## Make tables
Do all commands in `./`.

- Observations and results
``` 
python script/tab_phot_res.py data/gri_col.txt data/gri_2022GQ1_1s_N80.txt
```

## Plot figures
Do all commands in `./`.

- Ephemerides
``` 
# Obtain ephemerides from JPL
obtain_sssb_eph.py --obj "2021 UW1" "2021 TY14" "2022 GQ1" --date0 2021-08-01 --date1 2022-06-01 --step 1h --outdir data
# Plot
python script/plot_ephem.py  --f_ephem data/2021TY14_2021-08-01to2022-06-01step1hcode500eph.txt data/2021UW1_2021-08-01to2022-06-01step1hcode500eph.txt data/2022GQ1_2021-08-01to2022-06-01step1hcode500eph.txt  --f_mpc data/MPCobs_2021TY14_20251031.txt data/MPCobs_2021UW1_20251031.txt data/MPCobs_2022GQ1_20251031.txt
```

- Lightcurves and LS periodograms
``` 
python script/plot_lc.py data/gri_mag.txt data/gri_2022GQ1_1s_N80.txt
```

- Colors
```
# Color-color diagram
python script/plot_cc.py data/gri_col.txt

# Color lightcurve
python script/plot_colorlc.py data/gri_col.txt data/gri_2022GQ1_1s_N80.txt
``` 


- Phase curve
```
python script/plot_pc.py  --f_mpc data/MPCobs_2021TY14_20251031.txt data/MPCobs_2021UW1_20251031.txt data/MPCobs_2022GQ1_20251031.txt
```

- D-P
```
python script/plot_DP.py --Seimei --indivi
```


## Dependencies
This repository is depending on `Python`, `NumPy`, `pandas`, `SciPy`, `Astropy`, `Astroquery`.
