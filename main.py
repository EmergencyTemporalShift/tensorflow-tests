import numpy as np
from numpy import interp
import tensorflow as tf

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import math
from scipy import signal

import time

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from pylab import *

import sys # For arguments
#%matplotlib inline

from IPython.core.debugger import set_trace

import NetworkGroup as ng

# TODO: run tensorboard automatically without spamming console

#from tensorboard import main as tb

#import os
#import sys

#def launchTensorBoard(logdir, debug=False):
	#tf.flags.FLAGS.logdir = "./tensorboard_logs/" + logdir
	#if debug:
		#tf.flags.FLAGS.debugger_port = 6064
	#tb.main()

#launchTensorBoard('momentum')
#mode = None
if len(sys.argv):
	#print("No arguments") # Args don't work with ptpython and interactive, this always prints
	mode = 'nothing'
elif sys.argv[1] == 'momentum':
	mode = 'momentum'
elif sys.argv[1] == 'autoencoder':
	mode = 'autoencoder'
else:
	mode = 'nothing'
	
mode = 'general' # Hack! args aren't working with ptpython

if mode != 'nothing':
	print("Running {}".format(mode))
else:
	print("Not a valid mode")

if mode == 'momentum':
	n = ng.NetworkGroup('momentum', 2, 1, 1, mode='momentum')
elif mode == 'autoencoder':
	n = ng.NetworkGroup('autoencoder', 1, 5, 5, mode='autoencoder')
elif mode == 'adadelta':
	n = ng.NetworkGroup('adadelta', 4, 1, 1, mode='adadelta')
elif mode == 'general':
	n = ng.NetworkGroup('general', 5, 1, 1, mode='general')
else:
	n = None

if n:
	n.update_tensorboard()
	print("tensorboard" + " --logdir=./tensorboard_logs/" + str(mode))
	n.train_all(10000)
	n.update_tensorboard()

# run
# ptpython -i main.py
# or
# python -i main.py

#n.sess.run([n.nets[1].outputs], feed_dict={n.nets[1].inputs: np.expand_dims(np.array((.1,.2,.3,.4,.5)), axis=0)})


print("Interactive mode should be enabled now")
