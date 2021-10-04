import numpy as np
import torch

import csv

fp = open('label_map_manipulation.csv', 'r')
reader = csv.reader(fp)

modes = dict()
plotlis = []


for itr, info in enumerate(reader):
    if itr == 0:
        continue
    val = info[-1]
    plotlis.append(val)
    if val not in modes:
        modes[val] = 0

    modes[val] += 1

import matplotlib.pyplot as plt
# res = sorted(modes.items())

keys = modes.keys()
vals = modes.values()
print(modes)

fig, ax = plt.subplots()
ax.hist(plotlis, fill=True,bins=len(keys))

ax.set_title("Histogram for Cabin Left Hand")

ax.set_xlabel("Cabin Left Num")

labels = keys

rects = ax.patches

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2 , height+0.01, label,
            ha='center', va='bottom')
# plt.hist(plotlis, color ='r')
plt.savefig('result/histo2.png')
plt.show()



