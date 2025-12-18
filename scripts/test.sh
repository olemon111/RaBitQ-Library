#!/bin/bash

pushd /home/lbl/code/graduationDesign/rabitq/testqg/external/RaBitQ-Library
cmake -S test -B build-test -DCMAKE_BUILD_TYPE=Release
cmake --build build-test -j
./build-test/testqg
popd