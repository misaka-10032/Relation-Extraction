#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import sys

# print the accuracy
def print_name(file_name, name):
    x, train, valid = [], [], []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if name not in line: continue
            words = line.split(" ")
            x.append(int(words[1]))
            train.append(float(words[4]))
            valid.append(float(words[6]))

            if name == "cross_entropy":
               train = list(map(lambda x: -x, train))
               valid = list(map(lambda x: -x, valid))

    plt.plot(x[:100], train[:100], 'ro-', label="train")
    plt.plot(x[:100], valid[:100], 'x-', label="train")
    plt.xlabel('epochs', fontsize=14, fontname="Times New Roman")
    plt.ylabel(name, fontsize=14, fontname="Times New Roman")
    plt.show()

if __name__ == "__main__":
    print_name(sys.argv[1], str(sys.argv[2]))

