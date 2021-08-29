#!/bin/bash

[[ -e numerical_test_output.txt ]] && rm numerical_test_output.txt

for krylov_dim in 5 10 15 20 21 22 23 24 25 26 27 28 29 30 34 37 40 45 50
do
echo "Running for k = $krylov_dim"
./numerical_test -k $krylov_dim >> numerical_test_output.txt
done
