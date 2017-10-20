#!/bin/bash
#
#experimento=("_half_tissues_rf" "_empty_debris_rf" "_tumor_stroma_rf" "")
##
#for k in ${experimento[@]};
#do
	for i in 40 100 200 400;
	do
		for j in 1 2 3 4 5;
		do
			./classify_paciente.py "data/features_"$i".txt" "../filtro_8_classes/svm_tissues/dsfold"$j".txt" $i > "resultado"$j"_"$i".txt"
		done
	done
#done
#
