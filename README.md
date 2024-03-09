# Pathfinder

This github repo contains the code and scipts to generate prefetches using Pathfinder on SPEC and GAP traces, and generate results files that and generate result files that contain IPC numbers, LLC Load ACCESS, and LLC PREFETCH REQUEST, ISSUED to calculate accuracy and coverage numbers.

## Download datasets:

run 'download.sh' to download the traces we tested in our paper

## Create enviroment:
use this command 'conda env create -f environment.yml' to create enviroment for Pathfinder

## Building:
run './ml_prefetch_sim.py build' to compile

## Running and Evaluating:
run 'run_pathfinder_gap_spec.sh' to generate prefetch files in folder 'pathfinder_prefetches_gap_spec' and generate result files in results folder

