#!/bin/bash
#
for i in 40 100 200 400;
do
	./filtro.py "data/features_"$i".txt" "data/features_"$i"_tumor_stroma_rf.txt"
done
