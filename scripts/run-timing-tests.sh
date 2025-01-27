#!/bin/bash

# with cpu
# mpic++ -fopenmp -o ./timing ./matching-decomposition-timing.cc

# with gpu
nvcc -o timing matching-decomposition-timing.cu

N_TESTS=100

echo "nTors time"> timing-results.csv

for (( IDX=0; IDX<=N_TESTS; IDX++ )); do
    
    # with gpu
    ./timing >> ./timing-results.csv
    
    # with cpu
    # mpirun -np 10 ./timing >> ./timing-results.csv
done