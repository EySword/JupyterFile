import pandas as pd
import torch
import numpy as np
import json


def FindForce(outdata):

    f = []
    last = -1
    for x, line in enumerate(outdata):
        if "TOTAL-FORCE" in line:
            last = x+2
    for x, line in enumerate(outdata):
        if not x == last:
            continue
        if line[1] == "-":
            break
        last += 1
        f.append(list(map(float,line.split()[3:6])))

    return f


file_path='F:\\BiSeI_dis\\d0p05\\'


def get_force(start, end, file_path):
    '''
    ·µ»Ølist
    '''
    force_list=[]
    for i in range(start,end+1):
        if len(str(i))==1:
            zero_n = 3
        elif len(str(i))==2:
            zero_n = 2
        elif len(str(i))==3:
            zero_n = 1
        else:
            zero_n = 0
        out = open(file_path+"disp"+"0"*zero_n+str(i)+"\\OUTCAR")
#         print(file_path+"disp"+"0"*zero_n+str(i)+"\\OUTCAR")
        input_data = out.readlines()
        force = FindForce(input_data)
        out.close()
        force_list.append(force)
    return force_list


force_list = get_force(1,1000,file_path)


def saveJsonFile(python_list, filename):
    json_list = json.dumps(python_list)
    with open(filename, 'w') as file_obj:
      json.dump(json_list, file_obj)


saveJsonFile(force_list,"BiSeI_1000_force.json")


def loadJsonFile(filename):
    '''
    return python data structure
    '''
    with open(filename, 'r') as f:
        json_data = json.load(f)
    return json.loads(json_data)


l=loadJsonFile("BiSeI_1000_force.json")



