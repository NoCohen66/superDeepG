#!/bin/bash
# Reproduction script for Table2 - CIFAR10 (Sc1, IS=1.0)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/04_SC_1_PGD_CIFAR/04_table2_SC_1_CIFAR_PGD_254_1000.yaml
python analyze.py experiment/results/04_table2_SC_1_CIFAR_PGD_254_1000_results.csv
