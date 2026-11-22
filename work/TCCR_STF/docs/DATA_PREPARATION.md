# DATA_PREPARATION

## 1. LGC dataset (Coleambally→Goulburn? No — LGC = Lake Frome/Goulburn pair used by STFDiff)

Actually: LGC raw archives come as date folders containing a single 6-band
`.int` per sensor.

### ⚠️ CRITICAL: byte layout

| Sensor | File pattern | Layout | Shape |
|---|---|---|---|
| Landsat | `*_TM.int` | **BSQ** | (bands=6, rows=2720, cols=3200), uint16 LE |
| MODIS | `MOD09GA_A<YYYY>DDD.sur_refl.int` | **BSQ** | (bands=6, rows=2720, cols=3200), uint16 LE |

**Reading these files as BIL silently produces row-strided corrupted images.**
Symptom of the mistake: all 6 per-band means become identical. Use
`src/tccr/core.py::read_lgc_int_bsq`.

Band order after correct BSQ read (verified against spectral profile):
B1=blue, B2=green, B3=red, B4=NIR, B5=SWIR1, B6=SWIR2 (TM/ETM+ order, matches
MODIS bands reordered to Landsat as per the official pipeline).

### Preprocessing steps (tools/prepare_data.py)

1. Read with BSQ loader.
2. Clip DN to [0, 10000] (LGC has no out-of-range pixels; CIA does, ~2.6%).
3. Save per-date uint16 GeoTIFF-like TIFF (HWC).
4. Build triples (d1 < d2 < d3) → Group folders:
   `Group_XX/{Landsat_01,Landsat_02,Landsat_03,MODIS_01,MODIS_02,MODIS_03}/`
   Each patch 256×256×6, filename `Group_XX_<S>_<date>_r{row}c{col}.tif`.
5. Splits (frozen, from the official STFDiff experiments):
   - train: 5 triples (2004-04-16 … 2004-12-12)
   - val (holdout for selection): 2005-02-14 / 2005-03-02 / 2005-04-03
   - flood confirmation set (legacy; observed in earlier stages): 2004-12-28 /
     2005-01-13 / 2005-01-29 — name it **Legacy Flood Confirmation Set**, never
     "pristine test".

## 2. CIA dataset (Coleambally Irrigation Area, CSIRO DAP collection 6337)

| Sensor | File pattern | Layout | Shape |
|---|---|---|---|
| Landsat | `L71093084_*_HRF_modtran_surf_ref_agd66.bil` | **BIL** (genuine) | (rows=2040, cols=1720, bands=6) |
| MODIS | `MOD09GA_A<YYYY>DDD.sur_refl.int` | **BSQ** | (bands=6, rows=2040, cols=1720) |

- 17 common dates (2001-10 … 2002-05).
- Anomalous DN: ~2.6% of pixels >10000 (fill values up to 65535) → clip to
  [0, 10000]. **Do not skip this**: unclipped outliers dominate every
  correlation/statistic.
- Center-crop to 1792×1536, tile into 256×256 patches → 42 patches/date.
- Splits: date-level. With 15 consecutive triples, we recommend:
  calibration/train = first 8 triples, val = next 4, test = last 3
  (or leave-one-date-out CV if you prefer; freeze before running).

## 3. Fingerprints

`data_manifest_correct_bsq.csv` lists SHA256 of every source `.int`/`.bil`
consumed by the main experiments. Regenerate with
`python tools\fingerprint_data.py --raw <dir>`.

## 4. What NOT to do

- ❌ Do not read LGC as BIL (see `legacy_debug/` — kept only for the historical
  audit, marked DO NOT USE FOR TRAINING).
- ❌ Do not evaluate on CIA without clarifying which byte-layout reading you
  used; a mismatch is invisible in the images but destroys transfer.
- ❌ Do not fit calibration scalars on any test split.
