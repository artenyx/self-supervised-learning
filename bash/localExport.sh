#!/bin/bash
# Export ExperimentFiles to local downloads folder

TYPE=$1
AOC_DIR='geraldkwhite@aoc.uchicago.edu:/home/geraldkwhite/SSLProject/'
LOCAL_DIR='/Users/jerrywhite/Documents/01-UChicago/05-Thesis/01-ThesisExperiments/02-Exports'

if [ "$TYPE" = 'TO_LOCAL' ]; then
  SOURCE_DIR=AOC_DIR
  TARGET_DIR=LOCAL_DIR
elif [ "$TYPE" = 'FROM_LOCAL' ]; then
  SOURCE_DIR=LOCAL_DIR
  TARGET_DIR=AOC_DIR
else
  echo FALSE
fi

echo copy complete.
