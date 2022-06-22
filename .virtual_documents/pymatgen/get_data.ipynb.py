from pymatgen.ext.matproj import MPRester


with MPRester('WZW8U8nBTGJsz1i7WQ') as m:
    data = m.get_data("BiSeI")


data[0]['band_gap']


m


import json



