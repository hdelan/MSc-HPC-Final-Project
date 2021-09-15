This directory contains tests:

```
tests/linalg_test.cu
tests/SPMV_test.cu
tests/SPMV_blocks.cu
tests/lanczos_test.cu
```

- All tests use the blocksize found in ```lib/blocks.h```.
- All source files are found in ```lib/``` and ```tests/```.

```linalg_test``` Compares the speedups and relative error of CUDA vs serial implementations of dot product, norm, saxpy and reduce operations
  - To run use the command ```make linalg_test```

```SPMV_test``` Compares the speedups and relative error of four different CUDA implementations of SPMV vs serial.
  - To run use the command ```make SPMV_test```

```SPMV_blocks``` Runs the SPMV test for all blocksizes.
  - To run use the command ```make SPMV_blocks```

```lanczos_test``` runs the CUDA Lanczos version against the serial version and compares output. In the CUDA version all operations are performed in CUDA, including multOut.
  - To run use the command ```make lanczos_test```

