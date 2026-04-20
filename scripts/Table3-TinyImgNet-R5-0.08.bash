#!/bin/bash
# Reproduction script for Table3 - TinyImgNet (R5, IS=0.08)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_3_CGT_networks/Table_TinyImageNet10000/0_R_5_TINY_CROWNIBP_0p78125_10000.yaml
python analyze.py experiment/results/0_R_5_TINY_CROWNIBP_0p78125_10000_results.csv
