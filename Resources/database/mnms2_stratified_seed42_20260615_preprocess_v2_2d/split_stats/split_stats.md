# M&Ms-2 Split / Preprocess Summary

- split manifest: `parameters\mnms2_stratified_seed42_20260615_preprocess_v2_split.json`
- processed dataset: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d`

## Overview

| split | case_count | sample_count | slice_seen_count | ed_saved_slices | es_saved_slices | myo_effective_slices | empty_myo_slices_seen | topology_abnormal_slices_seen | topology_filtered_slices |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 216 | 3623 | 4838 | 1935 | 1688 | 3623 | 1215 | 531 | 0 |
| val | 72 | 1242 | 1678 | 661 | 581 | 1242 | 436 | 213 | 0 |
| test | 72 | 1229 | 1612 | 656 | 573 | 1229 | 383 | 170 | 0 |

## Distributions

| split | category | value | count |
| --- | --- | --- | --- |
| train | pathology | ARR | 21 |
| train | pathology | CIA | 21 |
| train | pathology | FALL | 21 |
| train | pathology | HCM | 36 |
| train | pathology | LV | 36 |
| train | pathology | NOR | 45 |
| train | pathology | RV | 18 |
| train | pathology | TRI | 18 |
| train | vendor | GE MEDICAL SYSTEMS | 32 |
| train | vendor | Philips Medical Systems | 53 |
| train | vendor | SIEMENS | 131 |
| train | field_strength | 1.5 | 206 |
| train | field_strength | 3 | 10 |
| val | pathology | ARR | 7 |
| val | pathology | CIA | 7 |
| val | pathology | FALL | 7 |
| val | pathology | HCM | 12 |
| val | pathology | LV | 12 |
| val | pathology | NOR | 15 |
| val | pathology | RV | 6 |
| val | pathology | TRI | 6 |
| val | vendor | GE MEDICAL SYSTEMS | 10 |
| val | vendor | Philips Medical Systems | 18 |
| val | vendor | SIEMENS | 44 |
| val | field_strength | 1.5 | 69 |
| val | field_strength | 3 | 3 |
| test | pathology | ARR | 7 |
| test | pathology | CIA | 7 |
| test | pathology | FALL | 7 |
| test | pathology | HCM | 12 |
| test | pathology | LV | 12 |
| test | pathology | NOR | 15 |
| test | pathology | RV | 6 |
| test | pathology | TRI | 6 |
| test | vendor | GE MEDICAL SYSTEMS | 11 |
| test | vendor | Philips Medical Systems | 17 |
| test | vendor | SIEMENS | 44 |
| test | field_strength | 1.5 | 68 |
| test | field_strength | 3 | 4 |

## Files

- overview csv: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\split_overview.csv`
- distribution csv: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\split_distributions.csv`
- case split csv: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\case_split.csv`
- sample records csv: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\sample_records.csv`
- slice records csv: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\slice_records.csv`
