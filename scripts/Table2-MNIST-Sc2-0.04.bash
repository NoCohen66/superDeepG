#!/bin/bash
# Reproduction script for Table2 - MNIST (Sc2, IS=0.04)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_2_PGD_networks/03_SC_2_PGD_MNIST/03_table2_SC_2_MNIST_PGD_10.yaml
python analyze.py experiment/results/03_table2_SC_2_MNIST_PGD_10_results.csv
