#!/bin/bash
# Reproduction script for Table2 - MNIST (R30, IS=6.0a)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/01_R_30_6_PGD_MNIST/01_table2_R_30_6_PGD_MNIST_14500.yaml
python analyze.py experiment/results/01_table2_R_30_6_PGD_MNIST_14500_results.csv
