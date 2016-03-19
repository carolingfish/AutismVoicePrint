#!/usr/local/bin/gnuplot

reset

set terminal png

# set your chart name here:
set output "predict.png"

set style data lines
set key right


###### Fields in the data file are
###### Inverse of regularization strength; Accurary(valid)

set title "Accurary(valid set) vs. Inverse of regularization strength"
set xlabel "Inverse of regularization strength"
set ylabel "Accurary(valid set)"

# set input file here:
plot "./both.csv" using 1:2 title 'Both', \
     "./male.csv" using 1:2 title 'Male', \
     "./female.csv" using 1:2 title 'Female'


reset
