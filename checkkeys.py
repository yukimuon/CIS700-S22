import numpy as np
from matplotlib import pyplot
import threading
import pickle
import argparse
import logging
import os
import random
import pprint

pp = pprint.PrettyPrinter(indent=4)

def print(x):
    pp.pprint(x)

attackdict = {}

for f in os.listdir():
    if f.endswith("pickle"):
        with open(f,'rb') as pf:
            obj = pickle.load(pf)
            attackdict.update(obj)

print(len(attackdict.keys()))
file_to_store = open(f"points.pickle", "wb")
pickle.dump(attackdict, file_to_store)
file_to_store.close()