#!/bin/bash
# Export ExperimentFiles to local downloads folder

TYPE=$1
SERVER_DIR='geraldkwhite@aoc.uchicago.edu:/home/geraldkwhite/SSLProject/ExperimentFile'
LOCAL_DIR='/Users/jerrywhite/Documents/01-UChicago/05-Thesis/01-ThesisExperiments/02-Exports'

[ "$TYPE" = 'TO_LOCAL' ] && scp -r $SERVER_DIR $LOCAL_DIR && STAT=$?
[ "$TYPE" = 'FROM_LOCAL' ] && scp -r $LOCAL_DIR $SERVER_DIR && STAT=$?
echo $STAT
[ $STAT -eq 0 ] && echo 'copy complete.'
[ $STAT -eq 1 ] && echo 'error with copy.'
