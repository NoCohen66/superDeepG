#!/bin/bash
# Reproduction script for Table2 - MNIST (T1, IS=0.2)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/07_T_1_02_PGD_MNIST/07_table2_T_1_02_PGD.yaml
python analyze.py experiment/results/07_table2_T_1_02_PGD_results.csv
