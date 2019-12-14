#!/bin/sh

#
# $ ./run.sh 2>&1 | tee -a result.log
#

set -e
set -x

run() {
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 1
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 2
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 4
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 8
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 16
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 32
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 64
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 128
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 256
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 512
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 768
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 1024
	./gl3cs_conv2d -x $1 -y $2 -z $3 -Z 2048

	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 1
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 2
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 4
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 8
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 16
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 32
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 64
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 128
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 256
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 512
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 768
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 1024
	./gl3cs_conv2d -x $1 -y $2 -z $3 -m -Z 2048
}


run 1 1 1

run 1 1 2
run 2 1 1
run 1 2 1

run 2 1 2
run 1 2 2
run 2 2 1
run 2 2 2

run 1 1 4
run 4 1 1
run 1 4 1

run 4 1 4
run 1 4 4
run 4 4 1
run 4 4 4

run 8 4 8

