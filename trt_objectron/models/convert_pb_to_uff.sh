#/bin/sh
set -e
set -x

python3 /usr/lib/python3.6/dist-packages/uff/bin/convert_to_uff.py object_detection_3d_chair.pb
