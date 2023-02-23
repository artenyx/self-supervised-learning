#!/bin/bash
# Export ExperimentFiles to local downloads folder

TYPE=$1
SERVER_DIR='geraldkwhite@aoc.uchicago.edu:/home/geraldkwhite/SSLProject/ExperimentFiles/ae_parallel-D-NL_02-14-2023_14-20-39cosreconalpha-0.001'
LOCAL_DIR='/Users/jerrywhite/Documents/01-UChicago/05-Thesis/01-ThesisExperiments/02-Exports'

STAT=1
[ "$TYPE" = 'FROM_LOCAL' ] && scp -r $LOCAL_DIR $SERVER_DIR && STAT=$?
[ "$TYPE" = 'TO_LOCAL' ] && scp -r $SERVER_DIR $LOCAL_DIR && STAT=$?
[ $STAT -eq 0 ] && echo 'copy complete.'
[ $STAT -eq 1 ] && echo 'error with copy.'
