#!/bin/bash
# Reproduction script for Table2 - CIFAR10 (Sh2, IS=0.02a)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/05_SH_2_PGD_CIFAR/05_table2_SH_2_CIFAR_PGD_2631_10.yaml
python analyze.py experiment/results/05_table2_SH_2_CIFAR_PGD_2631_10_results.csv
