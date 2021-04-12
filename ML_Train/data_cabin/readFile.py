import os 
import csv 
import torch
import shutil
import pandas as pd

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
def copyFile(file = "compile_mode.csv", targetPath = "./alldata"):
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

    reader = csv.reader(csvFile)
    cnt = 0
    
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
            png = png.rjust(5,'0')
            file_path = "./"+video_name+ "/Cabin/"+ png + ".png"
            if os.path.isfile(file_path) == False:
                file_path = "./"+video_name+ "/Cabin/cabin"+ png + ".png"
            if os.path.isfile(file_path) == False:
                file_path = "./"+video_name+ "/Cabin/scene"+ png + ".png"
            new_name = targetPath + "/" +str(cnt).rjust(5, '0') + ".png"
            cnt += 1
            true_label.append(mode)
            file_name.append(str(cnt).rjust(5, '0'))
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
            if itr % 2  == 0:
                partition.append('test')
            else:
                partition.append('val')

    print(len(partition))
    dataframe = pd.DataFrame({'filename':file_name, "numeric_label": true_label,"partition":partition, "Ruma": Ruma, "Nithin":Nithin,"Haomeng": Haomeng,
                "David":David, "Andrea": Andrea})
    dataframe.to_csv("label_map.csv")
    

    


if __name__ == "__main__":
    print('copy started')
    copyFile()