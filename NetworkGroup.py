import numpy as np
from numpy import interp
import tensorflow as tf


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import math
from scipy import signal

import time
from time import sleep
#from os import remove as removedir
import os, re

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#%matplotlib inline

from tensorflow.python import debug as tf_debug


import NetworkInstance as ni


class NetworkGroup(object):
	
	def __init__(self, log_dir, total_nets, num_inputs, num_outputs, mode=None):
		tf.reset_default_graph()
		self.sess = tf.Session()
		
		# This stuff is for debugging
		#self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
		#self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, 'localhost:6064')
		
		self.log_dir = log_dir
		self.total_nets = total_nets
		self.nets = {} # A dict of network classes
		
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
				
		self.current_net = 1 # I'm indexing my list of nets from 1 on purpose
		self.init_time = time.time() # Keeps tensorboard scalar plots from running on top of one another
		
		self.writer = None
		self.summaries = None
		
		self.update_tensorboard()
		

		
		if mode == 'momentum': # This broke somehow, don't know when
			self.train_group = 0
			self.assign_puts()
			
			self.init_momentumtest() # Momentum is broken
		
		if mode == 'autoencoder':
			self.random_some_ident(2)
			self.init_autoencoder([10,10,2,10,10]) # The outer bits determine the acuricy I think
		
		if mode == 'adadelta':
			self.train_group = 0
			self.assign_puts()
			
			self.init_adadeltatest()
			
		if mode == 'general':
			self.train_group = 0
			self.assign_puts()
			
			self.init_generaltest()
			
		self.init_nets()
		
		print("NetworkGroup Initialized")
		
	def init_nets(self):
		for _, net in self.nets.items():
			net.initialize()
		
	def init_autoencoder(self, mid_size):
		self.nets[1] = ni.NetworkInstance(self.sess, self.writer, 1, self.init_time, self.num_inputs, self.num_outputs, mid_size, learn_rate=1e-2, optimizer='adadelta', momentum=1e-2)
		
		
	def init_momentumtest(self):
		self.nets[1] = ni.NetworkInstance(self.sess, self.writer, 1, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='sgd', momentum=None)
		self.nets[2] = ni.NetworkInstance(self.sess, self.writer, 2, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='momentum', momentum=0.1)
		self.nets[3] = ni.NetworkInstance(self.sess, self.writer, 3, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='momentum', momentum=0.5)
		self.nets[4] = ni.NetworkInstance(self.sess, self.writer, 4, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='momentum', momentum=0.9)
	
	def init_adadeltatest(self):
		self.nets[1] = ni.NetworkInstance(self.sess, self.writer, 1, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='sgd')
		self.nets[2] = ni.NetworkInstance(self.sess, self.writer, 2, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=0.5e-1, optimizer='adadelta', rho=0.970, epsilon=1.0e-4)
		self.nets[3] = ni.NetworkInstance(self.sess, self.writer, 3, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='adadelta', rho=0.970, epsilon=1.0e-4)
		self.nets[4] = ni.NetworkInstance(self.sess, self.writer, 4, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1.5e-1, optimizer='adadelta', rho=0.970, epsilon=1.0e-4)
	
	def init_generaltest(self):
		self.nets[1] = ni.NetworkInstance(self.sess, self.writer, 1, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='sgd')
		self.nets[2] = ni.NetworkInstance(self.sess, self.writer, 2, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='momentum', momentum=0.9)
		self.nets[3] = ni.NetworkInstance(self.sess, self.writer, 3, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='adadelta', rho=0.970, epsilon=1.0e-4)
		self.nets[4] = ni.NetworkInstance(self.sess, self.writer, 4, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='adam', beta1=0.9, beta2=0.999, epsilon=1.0e-4)
		self.nets[5] = ni.NetworkInstance(self.sess, self.writer, 5, self.init_time, self.num_inputs, self.num_outputs, [4, 4], learn_rate=1e-1, optimizer='adam', beta1=0.9, beta2=0.999, epsilon=1.0e-8)
	
	def update_tensorboard(self):
		# Write down all the data for tensorboard to read
		#global writer
		self.writer = tf.summary.FileWriter('./tensorboard_logs/'+self.log_dir, self.sess.graph)
		#global summaries
		self.summaries = tf.summary.merge_all()
		self.writer.flush()
		
	def train(self, net_index, train_steps, spam=None, sleep=0):
		self.nets[net_index].train(train_steps, self.training_inputs, self.training_outputs, spam, sleep)
	
	def train_all(self, train_steps, spam=None, sleep=0):
		# How do I do this?
		# Should I interleave normal train with small steps?
		# If so I should pass an arg to reduce loss printouts.
		for i in self.nets:
			self.nets[i].train(train_steps, self.training_inputs, self.training_outputs, spam, sleep)
	
	def swap_active_net(net=None, reset=False):
		if net is None:
			current_net = 2 if current_net == 1 else 1
		else:
			current_net = net
		if reset:
			initialize_net(current_net)
		
		print("Current net is now: {}".format(current_net))

	def train_func(self, x):
		return self.square_wave(x)

	def square_wave(self, x):
		y = signal.square(3 * np.pi * 1.999 * x + np.pi)/2+0.5
		return y
	
	def random_ident(self, points=100):
		# Get a bunch of random numbers assigned to in and outs to train an autoencoder
		random = np.random.random_sample((points, self.num_inputs))
		self.training_inputs = random
		self.training_outputs = random
	
	def random_some_ident(self, ones=1, points=100):
		# One value is 1 the rest are zero
		values = np.zeros((points, self.num_inputs))
		#self.training_inputs  = np.zeros((points, self.num_inputs))
		#self.training_outputs = np.zeros((points, self.num_outputs))
		
		for i, v in enumerate(np.linspace(0, 1, points)):
			for _ in range(ones):
				high_val = np.random.randint(self.num_inputs)
				values[i][high_val] = 1
		
		self.training_inputs  = values
		self.training_outputs = values
	
	def assign_puts(self, points=100): # inputs and outputs
		self.training_inputs  = np.zeros((points, self.num_inputs))
		self.training_outputs = np.zeros((points, self.num_outputs))
		
		for i, v in enumerate(np.linspace(0, 1, points)):
			# Train group lets you append multiple runs
			self.training_inputs[i+self.train_group] = [v]
			self.training_outputs[i+self.train_group] = [self.train_func(v)]
		self.train_group += points
	
	def showGraph(self, detail=100, bounds=(0,1)):
		# Still needs work, especially for when there is multiple
		# inputs or networks
		print(self.nets)
		print(self.nets[1])
		x = np.zeros((detail)) # Do I need to set to zeros first?
		nets = np.zeros((self.total_nets, detail))
		for i, v in enumerate(np.linspace(bounds[0], bounds[1], detail)):
			# Set the X values for the graph
			x[i] = interp(i, [0,detail], [bounds[0], bounds[1]])
			# Set the values from the first net
			for j in range(self.total_nets):
				nets[j][i] = self.sess.run(self.nets[1].outputs, feed_dict={self.nets[1].inputs: np.array([[v]])})
		plt.plot(x, net[1])
		show()

	def manual_input(self, net, input_):
		# TODO: give a sensible error if input wrong size
		dinput = {}
		try:
			input_ = list(input_)
		except TypeError:
			print("The input needs to be listlike")
			return
		
		# Make sure the input has the right depth
		inlen = len(np.shape(input_)) # depth of input
		if inlen == 0: # 1
			dinput[0] = {}
			dinput[0][0] = input_
		elif inlen == 1: # [1,2,3,4]
			#temp = input_
			dinput[0] = input_
		elif inlen == 2: # [[1,2,3],[4,5,6],[7,8,9]]
			dinput = input_
		
		#print(dinput)
		
		for i, L in enumerate(dinput):
			assert len(L) == self.num_inputs, "Input has wrong size at index {}".format(i)
		
		output = []
		for i, L in enumerate(dinput):
			#print(i, L)
			#output[i] = {}
			output.append(self.sess.run(self.nets[net].outputs, feed_dict={self.nets[net].inputs: np.expand_dims(np.array(L), axis=0)}).tolist()[0])
			#print(round(temp, 2))
			
			# Do some rounding so it takes up less space
			for j in range(len(output[i])):
				output[i][j] = round(output[i][j], 2)
				
		return output
	
	def encode(self, input_, net_index, layer):
		"""Return the encoded value(s)"""
		return self.nets[net_index].layers[layer]['outputs'].eval(session=self.sess, feed_dict={self.nets[net_index].inputs: input_}).tolist()
	
	def decode(self):
		"""Reconstruct the original input"""
		# Probobly need to alter the computation graph
		pass
	
	def dump_nets(self):
		sess = self.sess
		for _, net in self.nets.items():
			print('\n%s' % net)
			for l in range(1, len(net.shape)):
				print('\nLayer %s' % l)
				print('biases')
				print(sess.run(net.layers[l]['biases']))
				print('weights')
				print(sess.run(net.layers[l]['weights']))

	def clear_tfevents(self, dir):
		dir = 'tensorboard_logs/' + dir
		for f in os.listdir( dir):
			if re.search('events.out.tfevents.*', f):
				os.remove(os.path.join(dir, f))
