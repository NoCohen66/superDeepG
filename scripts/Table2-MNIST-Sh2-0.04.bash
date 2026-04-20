#!/bin/bash
# Reproduction script for Table2 - MNIST (Sh2, IS=0.04)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/06_SH_2_PGD_MNIST/06_table2_C_SH_2_PGD_MNIST_63_10.yaml
python analyze.py experiment/results/06_table2_C_SH_2_PGD_MNIST_63_10_results.csv
