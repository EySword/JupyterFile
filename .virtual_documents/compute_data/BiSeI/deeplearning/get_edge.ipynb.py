import pandas as pd
import torch
import numpy as np
import json


def loadJsonFile(filename):
    '''
    return python data structure
    '''
    with open(filename, 'r') as f:
        json_data = json.load(f)
    return json.loads(json_data)


force_data=loadJsonFile("BiSeI_1000_force.json")



