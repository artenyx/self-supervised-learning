#!/bin/bash
# Run test to make sure package + repo installation successful and track export locations. Runs single AE for 1 epoch of rep learning and 1 epoch of le.

python main.py --usl_type ae_single --epochs_usl 1 --epochs_le 1 > exp.txt &