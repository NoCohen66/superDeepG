#!/bin/bash
# Reproduction script for Table6 - TinyImgNet (Sc0.02, IS=0.0002)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_6_TinyImageNet/sc2/tinyVNNCOMP_sc2_0p0002.yaml
python analyze.py experiment/results/tinyVNNCOMP_sc2_0p0002_results.csv
