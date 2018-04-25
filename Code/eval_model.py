import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf 
import numpy as np 
import os, sys, time, utils

path = '/home/jannis/workspace/Models/bas_g2p_r/normal _run_0'
num = 200

model = utils.retrieve_model(path,num)

# Do some predictions
