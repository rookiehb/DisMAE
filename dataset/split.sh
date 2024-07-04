#!/bin/bash

dir_name=("clip" "info" "paint" "quick" "real" "sketch")
pic_name=("The_Eiffel_Tower" "bee" "bird" "blueberry" "broccoli" "fish" "flower" "giraffe" "grass" "hamburger" "hexagon" "horse" "sun" "tiger" "toaster" "tornado" "train" "violin" "watermelon" "zigzag")

mkdir ./udg_dn/
for dir in ${dir_name[@]}
do
    mkdir ./udg_dn/$dir
    for pic in ${pic_name[@]}
    do
        # mkdir ./unsuperDN/$dir/$pic
        cd ./udg_dn/$dir
        ln -s ../../domain_net/$dir/$pic/ ./
        cd ../..
    done
done