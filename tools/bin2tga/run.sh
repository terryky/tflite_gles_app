#!/bin/sh
./bin2tga -f tflite_00_00_ssbo_bind0_ssbo1.bin -w 257 -c 3 -x
./bin2tga -f tflite_00_01_ssbo_bind0_ssbo2.bin -w 257 -c 4 -x
./bin2tga -f tflite_00_02_ssbo_bind0_ssbo8.bin -w 129 -c 4 -x -H
./bin2tga -f tflite_00_03_ssbo_bind0_ssbo9.bin -w 129 -c 4 -x -H
./bin2tga -f tflite_00_04_ssbo_bind0_ssbo8.bin -w 129 -c 4 -x -H
./bin2tga -f tflite_00_05_ssbo_bind0_ssbo9.bin -w 65 -c 4 -x -H
