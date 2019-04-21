#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_file', type=str, default='', help='input to extracted postprocessed file **.txt-post')

args=parser.parse_args()

input_file=args.input_file

if input_file=='':
    print('usage ./program --input_file=xxx.txt-post')
    sys.exit(1)

with open(input_file, 'r') as f:
    for line in f:
        line=line.strip()
        if line.startswith('#'):
            print(line)
            continue
        arr=line.split()
        if float(arr[-1])>0:
            continue
        print(line)
