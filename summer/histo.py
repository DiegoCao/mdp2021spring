import csv

filepath = "./Yolo_singlefile.csv"

import matplotlib.pyplot as plt
def plotDiffhisto(data):
    diff = []
    for i in range(len(data)-1):
        diff.append(abs(data[i]-data[i+1]))
    
    print('the max value of data is :', max(data))
    plt.hist(diff, bins = 'auto')
    plt.xlabel('wdata')
    plt.ylabel('count')
    plt.savefig('whisto.png')
    

x_data = []
y_data = []
w_data = []
h_data = []
with open(filepath) as fp:
    data = csv.reader(fp)
    itr = 0
    for row in data:


        if itr > 0:
            
            x_data.append(float(row[6]))
            y_data.append(float(row[7]))
            w_data.append(float(row[8]))
            h_data.append(float(row[9]))

        itr += 1


plotDiffhisto(w_data)
    
