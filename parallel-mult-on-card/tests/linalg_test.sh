#!/bin/bash

[[ -e linalg_test_output.txt ]] && rm linalg_test_output.txt

for block_size in 4 8 16 32 64 128 256 512 1024
do
echo "Making for blocksize $block_size"
echo "#define BLOCKSIZE $block_size" > ./lib/blocks.h
make linalg_test
echo "Running for blocksize $block_size"
./linalg_test >> linalg_test_output.txt
done
