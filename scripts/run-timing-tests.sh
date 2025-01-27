#!/bin/bash

# with cpu
g++ -std=c++11 -O3 -march=native -funroll-loops -o ./timing-cpu ./matching-decomposition-timing.cc

# with gpu
nvcc -o timing-gpu matching-decomposition-timing.cu

N_TESTS=100

echo "nTors time"> timing-results-cpu.csv
echo "nTors time"> timing-results-gpu.csv

for (( IDX=0; IDX<=N_TESTS; IDX++ )); do

    ./timing-cpu >> ./timing-results-cpu.csv

    ./timing-gpu >> ./timing-results-gpu.csv

done