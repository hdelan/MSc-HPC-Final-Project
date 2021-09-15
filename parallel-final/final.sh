#!/bin/bash

[[ -e final_output.txt ]] && rm final_output.txt 

for krylov_dim in 30
do
for file in bn1000000e9999944 adaptive California channel-500x100x100-b050 com-LiveJournal coPapersDBLP  NotreDame_yeast hugetrace-00020 road_central delaunay_n24 #europe_osm
do  
  echo "$file krylov dim $krylov_dim" >> final_output.txt
  ./final -k $krylov_dim -f $file >> final_output.txt
done
done
