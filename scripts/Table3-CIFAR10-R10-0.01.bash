#!/bin/bash
# Reproduction script for Table3 - CIFAR10 (R10, IS=0.01)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_3_CGT_networks/1_R_10_CIFAR.yaml
python analyze.py experiment/results/1_R_10_CIFAR_results.csv
