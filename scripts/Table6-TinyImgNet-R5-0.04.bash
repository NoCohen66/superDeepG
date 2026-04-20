#!/bin/bash
# Reproduction script for Table6 - TinyImgNet (R5, IS=0.04)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_6_TinyImageNet/r5/tinyVNNCOMP_r5_0p04.yaml
python analyze.py experiment/results/tinyVNNCOMP_r5_0p04_results.csv
