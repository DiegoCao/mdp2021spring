import os
import shutil
import csv

def copy_File():
    pass


if __name__ == "__main__":
    fp = open("label_map.csv")
    reader = csv.reader(fp)
    dataset = set()
    for i, j in enumerate(reader):
        if i == 0:
            continue
        dataset.add(j[2])
    print(dataset)
