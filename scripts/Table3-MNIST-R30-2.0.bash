#!/bin/bash
# Reproduction script for Table3 - MNIST (R30, IS=2.0)
python main_batch.py --config_path experiment/specification/tables_specifications/Table_3_CGT_networks/2_R_30_MNIST.yaml
python analyze.py experiment/results/2_R_30_MNIST_results.csv
