import pandas as pd
import torch
import numpy as np
import json
import pymatgen.core.structure
from pymatgen.io.vasp import Poscar
import pymatgen.core.periodic_table as pt


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


def getPoscarNeighbors(filename, bond_range = 3.5):
    '''
    return python list of all bonds
    '''
    poscar = Poscar.from_file(filename).structure
    nbs = poscar.get_all_neighbors(bond_range)
    edges=[[],[]]
    for i,nb in enumerate(nbs):
        for j in nb:
            edges[0].append(i)
            edges[1].append(int(j.index))
    return edges