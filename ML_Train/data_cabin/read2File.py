import os 
import csv 
import torch
import shutil
import pandas as pd
import matplotlib.pyplot as plt
def readCsv(file = "Compile_.csv"):
    csvFile = open(file, "r")
    reader = csv.reader(csvFile)
    Video_names = set()
    for i, j in enumerate(reader):
        if i==0:
            continue
        Video_names.add(j[0])
        video_set.append(j[0])
    print(Video_names)
    
    return Video_names, video_set

import random 
def copyFile(file = "/Users/hangruicao/Documents/mdp/mdp2021spring/ML_Train/V5_CabinCoding_CompiledData - Manipulation.csv", targetPath = "./dupdata"):
    csvFile = open(file, "r")
    if os.path.exists(targetPath)== False:
        os.makedirs(targetPath)

    file_name = []
    true_label = []
    numeric_label = []

    reader = csv.reader(csvFile)
    cnt = 0
    dict_cnt = {}
    labeldict = {}
    labelcnt = 0
   
    
    for itr, info in enumerate(reader):
        if itr == 0:
            continue
        else:
            tmplis = []
            video_name = info[0]
            png = info[1]
            print(cnt)
            if info[7] == "#N/A":
                continue
                
            tmplis.append(int(info[2]))
            tmplis.append(int(info[3]))
            tmplis.append(int(info[4]))
            tmplis.append(int(info[5]))
            tmplis.append(int(info[6]))
            for onelabel in tmplis:
                png = png.rjust(5,'0')
            # if video_name == "V_039_0027_290170":
            #     break
                file_path = "../Cabin_"+video_name+ "/Cabin/"+ png + ".png"
                if os.path.isfile(file_path) == False:
                    file_path = "../Cabin_"+video_name+ "/Cabin/cabin"+ png + ".png"
                if os.path.isfile(file_path) == False:
                    file_path = "../Cabin_"+video_name+ "/Cabin/scene"+ png + ".png"
            
                new_name = targetPath + "/" +str(cnt).rjust(5, '0') + ".png"

                true_label.append(onelabel)
                if onelabel not in labeldict:
                    labeldict[onelabel] = labelcnt
                    labelcnt += 1
                numeric_label.append(labeldict[onelabel])

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


    print('the length', len(file_name), ' ', len(true_label), ' ', len(partition))
    dataframe = pd.DataFrame({'filename':file_name, "numeric_label": numeric_label, "semantic_label": true_label, 
                "partition" :partition})
    dataframe.to_csv("duplicate2_manipulation.csv")
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