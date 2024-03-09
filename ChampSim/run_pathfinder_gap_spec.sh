#!/bin/bash

mkdir -p pathfinder_prefetches_gap_spec

# script to run pathfinder on gap and spec
find "./gap_spec_traces" -type f -name "*.txt.xz" | while read xz; do
  file_variable="$(basename $xz .txt.xz)"
  file_cc="cc-5"
  file_605mcf="605.mcf-s1"
  file_bfs10="bfs-10"
  file_450soplex="450.soplex-s0"
  file_623xalan="623.xalancbmk-s1"
  file_471omnetpp="471.omnetpp-s1"
  file_482sphinx3="482.sphinx3-s0"
  file_473astar="473.astar-s1"


  ./ml_prefetch_sim.py generate "$xz" ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$(basename "$xz" .txt.xz)".txt --model temp

  if [[ "$file_variable" == "$file_cc" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 20700644
  fi

  if [[ "$file_variable" == "$file_605mcf" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 38467912
  fi

  if [[ "$file_variable" == "$file_bfs10" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 61219343
  fi

  if [[ "$file_variable" == "$file_450soplex" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 29478984
  fi

  if [[ "$file_variable" == "$file_623xalan" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 52596273
  fi

  if [[ "$file_variable" == "$file_471omnetpp" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 54612142
  fi

  if [[ "$file_variable" == "$file_482sphinx3" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 84756213
  fi

  if [[ "$file_variable" == "$file_473astar" ]]; then
    ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 98033975
  fi

done

