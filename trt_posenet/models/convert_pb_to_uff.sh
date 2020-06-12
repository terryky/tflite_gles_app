#/bin/sh
set -e
set -x

python3 /usr/lib/python3.6/dist-packages/uff/bin/convert_to_uff.py posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.pb
