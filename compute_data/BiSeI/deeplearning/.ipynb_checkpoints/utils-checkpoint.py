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

def saveJsonFile(python_list, filename):
    '''
    save python structure as json file
    '''
    json_list = json.dumps(python_list)
    with open(filename, 'w') as file_obj:
      json.dump(json_list, file_obj)