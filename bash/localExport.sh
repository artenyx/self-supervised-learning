#!/bin/bash
# Export ExperimentFiles to local downloads folder

TYPE=$1
SERVER_DIR='geraldkwhite@aoc.uchicago.edu:/home/geraldkwhite/SSLProject/ExperimentFiles'
LOCAL_DIR='/Users/jerrywhite/Documents/01-UChicago/05-Thesis/01-ThesisExperiments/02-Exports'

if [ "$TYPE" = 'TO_LOCAL' ]; then
  SOURCE_DIR=SERVER_DIR
  TARGET_DIR=LOCAL_DIR
elif [ "$TYPE" = 'FROM_LOCAL' ]; then
  SOURCE_DIR=LOCAL_DIR
  TARGET_DIR=SERVER_DIR
else
  echo FALSE
fi

[ "$TYPE" = 'TO_LOCAL' ] && scp -r $SERVER_DIR $LOCAL_DIR
[ "$TYPE" = 'FROM_LOCAL' ] && scp -r $LOCAL_DIR $SERVER_DIR
echo 'copy complete.'
