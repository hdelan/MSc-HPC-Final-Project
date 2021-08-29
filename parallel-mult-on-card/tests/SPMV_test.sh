#!/bin/bash

[[ -e SPMV_test_output.txt ]] && rm SPMV_test_output.txt

for block_size in 4 8 16 32 64 128 256 512 1024
do
echo "Making for blocksize $block_size"
echo "#define BLOCKSIZE $block_size" > ./lib/blocks.h
make SPMV_test
echo "Running for blocksize $block_size"
./SPMV_test >> SPMV_test_output.txt
done
