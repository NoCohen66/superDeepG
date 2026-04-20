#!/bin/bash
# Reproduction script for Table2 - MNIST (R30, IS=1.0)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/00_R_30_1_PGD_MNIST/00_table2_R_30_1_PGD_MNIST_255.yaml
python analyze.py experiment/results/00_table2_R_30_1_PGD_MNIST_255_results.csv
