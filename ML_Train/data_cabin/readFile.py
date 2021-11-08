import os 
import csv 
import torch
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def readCsv(file = "Compile_.csv"):
    csvFile = open(file, "r")
    reader = csv.reader(csvFile)
    Video_names = set()
    video_set = []
    for i, j in enumerate(reader):
        if i==0:
            continue
        Video_names.add(j[0])
        video_set.append(j[0])
    print(Video_names)
    
    return Video_names, video_set

import random 
def copyFile(file = "V5_CabinCoding_CompiledData - Manipulation.csv", targetPath = "./alldata"):
    csvFile = open(file, "r")
    if os.path.exists(targetPath)== False:
        os.makedirs(targetPath)

    file_name = []
    true_label = []
    Ruma = []
    Nithin = []
    Haomeng = []
    David = []
    Andrea = []
    numeric_label = []

    reader = csv.reader(csvFile)
    cnt = 0
    lis = [1, 2, 3, 10]
    labeldict = {}
    labelcnt = 0
    for itr, info in enumerate(reader):
        if itr == 0:
            continue
        else:
            video_name = info[0]
            png = info[1]
            print(itr)
            print(info[7])
            if info[7] == "#N/A":
                continue
                
            mode = int(info[7])
            Ruma.append(int(info[2]))
            Nithin.append(int(info[3]))
            Haomeng.append(int(info[4]))
            David.append(int(info[5]))
            Andrea.append(int(info[6]))
            labelarr = np.array([int(info[i]) for i in range(2, 7)])
            probvec = np.array([(labelarr == int(lis[i])).sum()*(1/5) for i in range(0, 4)])

            summ = sum(probvec)
            if summ < 1:
                print('ERROR')
                for i in range(2, 7):
                    print(info[i])
                exit(1)



            
            png = png.rjust(5,'0')
            # if video_name == "V_039_0027_290170":
            #     break
            file_path = "../Cabin_"+video_name+ "/Cabin/"+ png + ".png"
            if os.path.isfile(file_path) == False:
                file_path = "../Cabin_"+video_name+ "/Cabin/cabin"+ png + ".png"
            if os.path.isfile(file_path) == False:
                file_path = "../Cabin_"+video_name+ "/Cabin/scene"+ png + ".png"
            
            new_name = targetPath + "/" +str(cnt).rjust(5, '0') + ".png"

            true_label.append(probvec)
            # if mode not in dict_cnt:
            #     dict_cnt[mode] = 0
            # dict_cnt[mode]+=1

            if mode not in labeldict:
                labeldict[mode] = labelcnt
                labelcnt += 1
            numeric_label.append(labeldict[mode])
            tname = str(cnt).rjust(5, '0')+'.png'
            cnt += 1
            file_name.append(tname)
            shutil.copyfile(file_path, new_name)

    Total_num = len(file_name)
    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
    train_num = int(train_ratio*Total_num)
    sample = random.sample(file_name, train_num)
    partition = []
    itr = 0
    for name in file_name:
        if name in sample:
            partition.append('train')
        else:
            itr += 1
            if itr % 2  == 0:
                partition.append('test')
            else:
                partition.append('val')


    print('the length', len(file_name), ' ', len(true_label), ' ', len(partition), len(Ruma), len(Andrea), len(Nithin), len(David), len(Andrea))
    dataframe = pd.DataFrame({'filename':file_name, "numeric_label": true_label, "semantic_label": true_label, 
                "partition" :partition, "Ruma": Ruma, "Nithin":Nithin,"Haomeng": Haomeng,
                "David":David, "Andrea": Andrea})
    dataframe.to_csv("prob_map_manipulation.csv")
    # x = dict_cnt.keys()
    # y = dict_cnt.values()
    # plt.hist(y)
    # plt.xlabel('manipulation mode')
    # plt.ylabel('cnt')
    # plt.savefig('manipulation')
    # plt.show()

if __name__ == "__main__":
    print('copy started')
    copyFile()