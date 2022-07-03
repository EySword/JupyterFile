import utils
import pymatgen.core.structure
from pymatgen.io.vasp import Poscar
import numpy as np


PT=utils.loadJsonFile('periodic_table.json')


poscar = Poscar.from_file("CONTCAR")


s=poscar.structure


s.as_dataframe()


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


edge_list = getPoscarNeighbors('CONTCAR')


np.array(edge_list)


list(s.cart_coords[0])


def getElementsNodes(filename):
    '''
    return python list of all element features
    '''
    poscar = Poscar.from_file(filename).structure
    ele = poscar.atomic_numbers
    poss = poscar.cart_coords
    ele_num = len(ele)
    nodes=[]
    for i in range(ele_num):
        node = []
        node = node + (list(poss[i]))
        nodes.append(node)
    return nodes


node_list = getElementsNodes('CONTCAR')


node_list


import pymatgen.core.periodic_table as pt


def elementFeatures(ele_name):
    ele = pt.Element(ele_name)
