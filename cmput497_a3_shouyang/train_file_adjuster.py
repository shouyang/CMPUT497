# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:07:20 2019

This file converts the A3DataCleaned files into underscore suffixed
one-sentence per line txt files. This prints the adjusted text to stdout.

@author: Daniel Zhou
"""

import sys

input_file = open(sys.argv[1], "r").read().split("\n")

output = []
for idx, line in enumerate(input_file):

    if line:
        word = line.split(" ")[0]
        tag  = line.split(" ")[-1]

        output.append(word + "_" + tag)
        
    else:
        print(" ".join(output))
        output = []
