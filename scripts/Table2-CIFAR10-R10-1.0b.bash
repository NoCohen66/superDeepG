#!/bin/bash
# Reproduction script for Table2 - CIFAR10 (R10, IS=1.0b)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/02_R_10_1_PGD_CIFAR/02_table2_R_10_1_PGD_CIFAR_2000_1000.yaml
python analyze.py experiment/results/02_table2_R_10_1_PGD_CIFAR_2000_1000_results.csv
