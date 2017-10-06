#!/bin/bash
#
for i in 40 100 200 400;
do
	./filtro.py "data/features_"$i".txt" "data/features_"$i"_half_tissues_rf.txt"
done
