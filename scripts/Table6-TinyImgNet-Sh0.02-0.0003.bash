#!/bin/bash
# Reproduction script for Table6 - TinyImgNet (Sh0.02, IS=0.0003)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_6_TinyImageNet/sh2/tinyVNNCOMP_sh2_0p0003.yaml
python analyze.py experiment/results/tinyVNNCOMP_sh2_0p0003_results.csv
