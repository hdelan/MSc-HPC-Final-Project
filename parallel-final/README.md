This directory contains the final implementation of the parallel Lanczos method.

To compile the code run ```make```. 

To run the code use:

```
./final -f [file] -k [krylov_dim]
```

A note on files: 
  - Files are not addressed by their path, but rather their name. So to compute on the file ../data/NotreDame_yeast/NotreDame_yeast.mtx 

  We use the command:
    ```./final -f NotreDame_yeast ```

To run Lanczos on all matrices run ```./final.sh``` which will produce output to ```final_output.txt```.

To change the Lanczos method to single precision, uncomment the lines 103 & 108 and comment the lines 104 & 109.
