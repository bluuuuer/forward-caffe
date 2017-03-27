#!/bin/bash

clear
echo "make ..."
cd build
make -j8
cd ..
echo "Run build/bin/jaffe_test"
./build/bin/jaffe_test
