#!/bin/bash
# Reproduction script for Table3 - TinyImgNet (Sc0.02, IS=0.001)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_3_CGT_networks/Table_TinyImageNet10000/2_SC_2_TINY_CROWNIBP_1000_001_10000im.yaml
python analyze.py experiment/results/2_SC_2_TINY_CROWNIBP_1000_001_10000im_results.csv