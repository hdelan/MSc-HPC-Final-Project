# Serial Lanczos Method

To run a basic example of code, use the command ```make run```

To run the Lanczos approximation for any matrix run:
```
./adj -f [../data/somedir/somefile.mtx] -k [krylov_dim]
```
Any ```.mtx``` file can be run, provided any comments are removed from the ```.mtx``` file. The graph must also be unweighted, meaning there are only two columns in the ```.mtx``` file. 

## Tests

There are two numerical tests:

- ```make numerical_test``` will compare the Lanczos approximation of e^Ax with a true analytic version of e^Ax for the matrix NotreDame_yeast.

- ```make numerical_test_orthog``` will run the Lanczos approximation with additional Arnoldi routines, and compare this answer with a true analytic version of e^Ax. Once again using the matrix NotreDame_yeast.

An additional reorthogonalize routine can be turned on in tests/numerical_test_orthog.cc by changing the variable ```bool reorthogonalize``` to ```true```, although this results in terrible innacuracy.

Find the code for these tests in ```tests/```

The methods can be trialled for many different krylov dimensions by running ```./tests/numerical_test.sh``` or ```./tests/numerical_test_orthog.sh``` and observing the output, which will be entered into a new ```.txt``` file.

