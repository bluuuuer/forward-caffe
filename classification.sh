#!/bin/bash

clear
cd build
echo "make ..."
make -j16
cd ..
echo "Run build/bin/classification_test ..."
for ((i = 0; i < 10; i++));
do
./build/bin/classification_test \
res/classification/deploy.prototxt \
res/classification/bvlc_reference_caffenet.caffemodel \
res/classification/imagenet_mean.binaryproto \
res/classification/synset_words.txt \
0 \
res/classification/cat.jpg
done
