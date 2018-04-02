#! /bin/bash

set -e

size=$1
src=$2
tgt=$3
prefix=$4
prefix=data/$4

real_size=$(($size * 1000))
folder=$prefix-${size}K
if [ -d $folder  ]; then
    echo $folder already exists
    exit
fi

mkdir $folder
cp $prefix/dev* $folder
cp $prefix/test* $folder
#python scripts/shuffle.py $prefix/train.$src $prefix/train.$tgt
shuffled_src=$prefix/train.${src}.shuffled
shuffled_tgt=$prefix/train.${tgt}.shuffled
if [ -f $shuffled_src ]; then 
    echo shuffled files already exist
else
    paste -d '|' $prefix/train.$src $prefix/train.$tgt | shuf | awk -v FS="|" '{ print $1 > "out_src" ; print $2 > "out_tgt" }'
    mv out_src $shuffled_src
    mv out_tgt $shuffled_tgt
fi

head -n $real_size $shuffled_src > $folder/train.$src
head -n $real_size $shuffled_tgt > $folder/train.$tgt
cp $folder/train.$src fast_align
cp $folder/train.$tgt fast_align


# train + dev
#cp $folder/dev.$src fast_align
#cp $folder/dev.$tgt fast_align
#cat fast_align/train.$src fast_align/dev.$src > fast_align/cat.$src
#cat fast_align/train.$tgt fast_align/dev.$tgt > fast_align/cat.$tgt
#python scripts/combine.py fast_align/cat.$src fast_align/cat.$tgt


# train only
python utils/vocab.py -dp $folder/train.$src --size 0 -vp $folder/vocab.$src -t 1
python utils/vocab.py -dp $folder/train.$tgt --size 0 -vp $folder/vocab.$tgt -t 1

