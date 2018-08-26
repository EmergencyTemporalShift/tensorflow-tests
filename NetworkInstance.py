import numpy as np
from numpy import interp
import tensorflow as tf

import time
from time import sleep

import re

class NetworkInstance(object):
	
	def __init__(self, session, writer, net_index, init_time, num_inputs, num_outputs, inner_shape, learn_rate=None, optimizer=None, momentum=None, rho=None, beta1=None, beta2=None, epsilon=None):
		# Maybe I should pass the networkInstance in 
		self.sess = session
		self.writer = writer
		self.log_dir = writer.get_logdir()
		self.init_time = init_time
		self.net_index = net_index
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		assert type(inner_shape) is list
		self.shape = [num_inputs]+inner_shape+[num_outputs]
		self.num_layers = len(self.shape)
		self.layers = {}
		self.inputs = None
		self.outputs = None
		self.init = {}
		#self.train
		self.learn_rate = learn_rate
		self.optimizer = optimizer
		self.momentum = momentum
		if self.optimizer == 'adadelta':
			if rho is None:
				rho = 0.95
			if epsilon is None:
				epsilon = 1e-8
		if self.optimizer == 'adam':
			if beta1 == None:
				beta1 = 0.9
			if beta2 is None:
				beta2 = 0.999
			if epsilon is None:
				epsilon = 1e-8
		self.rho = rho
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		if self.optimizer == 'sgd':
			self.setup_net(learn_rate=self.learn_rate, optimizer=self.optimizer)
		if self.optimizer == 'momentum':
			self.setup_net(learn_rate=self.learn_rate, optimizer=self.optimizer, momentum=self.momentum)
		if self.optimizer == 'adadelta':
			self.setup_net(learn_rate=self.learn_rate, optimizer=self.optimizer, rho=self.rho, epsilon=self.epsilon)
		if self.optimizer == 'adam':
			self.setup_net(learn_rate=self.learn_rate, optimizer=self.optimizer, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)

	def __repr__(self):
		return("<Network Instance No{}>".format(self.net_index))
	
	def netinfo(self):
		print("Network {}".format(self.net_index))
		print("Network Optimizer: {}".format(self.optimizer))
		print("Learn rate: {}".format(self.learn_rate))
		
		if self.optimizer == 'momentum':
			print("Uses momentum")
			print("Momentum: {}".format(self.momentum))
		#else:
			#print("Doesn't use momentum")
		if self.optimizer == 'adadelta':
			print("rho: {}".format(self.rho))
			print("epsilon: {}".format(self.epsilon))
		if self.optimizer == 'adam':
			print("beta1: {}".format(self.beta1))
			print("beta2: {}".format(self.beta2))
			print("epsilon: {}".format(self.epsilon))
		print()
		
	
	def apply_weights_add_bias_take_sigmoid(self, inputs, weights, biases, layer_index):
		with tf.name_scope('output'):
			matmul = tf.matmul(inputs, weights, name='apply_weights')
			add = tf.add(matmul, biases, name = 'add_bias')
			return tf.nn.sigmoid(add, name = 'take_sigmoid')

	def error_function(self, logits, desired_outputs):
		with tf.name_scope('error_function'):
			sub = tf.subtract(logits, desired_outputs, name='difference_of_logits_desired')
			square = tf.square(sub, name='square')
			reduce = tf.reduce_sum(square, name= 'reduce')
			return tf.multiply(0.5, reduce, name='error_function')
		
	
	
	def make_layer(self, layer_index, inputs, a, b, isFinal=False):
		with tf.variable_scope("layer_{}".format(layer_index), reuse=tf.AUTO_REUSE):
			self.layers[layer_index] = {} # Initializes a layer dictionary
			
			layer = self.layers[layer_index] # Makes a shortcut
			
			layer['weights'] = tf.get_variable('weights', shape=[a,b], initializer=tf.initializers.truncated_normal())
			layer['biases']  = tf.get_variable('biases',  shape=[b],   initializer=tf.initializers.zeros())
			
			layer['outputs'] = self.apply_weights_add_bias_take_sigmoid(inputs, layer['weights'], layer['biases'], layer_index)
		if isFinal:
			# This makes the final output have a node outside the layer scope
			self.outputs = tf.identity(layer['outputs'], name = 'outputs')
	
	def make_layers(self, sizes):
		for tmp, size in enumerate(sizes):
			idx = tmp+1
			if tmp == 0: # First layer
				self.make_layer(1, self.inputs, sizes[tmp], sizes[idx])
			elif tmp < len(sizes)-2: # Middle layer
				self.make_layer(idx, self.layers[tmp]['outputs'], sizes[tmp], sizes[idx])
			elif tmp == len(sizes)-2: # Final layer
				self.make_layer(idx, self.layers[tmp]['outputs'], sizes[tmp], sizes[idx], isFinal=True)
	
	def setup_net(self, learn_rate=1e-1, optimizer='sgd', momentum=1e-1, rho=0.95, beta1=0.9, beta2=0.999, epsilon=1e-8):
		"""Set up but don't initialize yet"""
		
		if optimizer is None:
			optimizer = 'sgd'
		if learn_rate is None:
			learn_rate = 1e-1
		if momentum is None:
			momentum = 1e-1
			
		with tf.variable_scope('network_{}'.format(self.net_index), reuse=tf.AUTO_REUSE):
			# Batch input
			self.inputs = tf.placeholder(tf.float32, shape=[None, self.num_inputs], name = 'inputs')
			# Batch output
			self.desired_outputs = tf.placeholder(tf.float32, shape=[None, self.num_outputs], name='desired_outputs')
			
			self.make_layers(self.shape)
			
			self.error_tensor = self.error_function(self.outputs, self.desired_outputs)
			
			if optimizer == 'sgd':
				self.train_step = tf.train.GradientDescentOptimizer(learn_rate, name='train_step').minimize(self.error_tensor)
			elif optimizer == 'momentum':
				self.train_step = tf.train.MomentumOptimizer(learn_rate, momentum, name='momentum_train_step').minimize(self.error_tensor)
			elif optimizer == 'adadelta':
				self.train_step = tf.train.AdadeltaOptimizer(learn_rate, rho=rho, epsilon=epsilon, name='adadelta_train_step').minimize(self.error_tensor)
			elif optimizer == 'adam':
				self.train_step = tf.train.AdamOptimizer(learn_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, name='adam_train_step').minimize(self.error_tensor)
		
		with tf.variable_scope('network_{}/loss_ops'.format(self.net_index), reuse=tf.AUTO_REUSE):
			# Honestly might not be needed anymore
			self.loss = tf.get_variable('loss', shape=[], initializer=tf.initializers.constant(0))
			self.loss_assign = tf.assign(self.loss, self.error_tensor, name='loss_assign')
		
		self.init['defined'] = False
		
		with self.sess.as_default():
			self.summary = tf.summary.scalar('run_{}/network_{}/loss'.format(self.init_time, self.net_index), self.error_tensor)
	
	def setup_init(self, weightbias):
		self.init['loss']  = tf.variables_initializer([self.loss])
		# Others is for weights and biases
		self.init['weightbias'] = tf.variables_initializer(weightbias, name='network_{}_weightbias_init'.format(self.net_index))
		
		if self.optimizer == 'momentum':
			self.init['momentum'] = self.initialize_search('momentum_train_step')
		elif self.optimizer == 'adadelta':
			self.init['adadelta'] = self.initialize_search('adadelta_train_step')
		elif self.optimizer == 'adam':
			self.init['adam'] = self.initialize_search(['adam_train_step', 'beta'])
		
		self.init['defined'] = True
	
	
	def initialize(self, debug_other=False):
		print("Initializing net {}".format(self.net_index))
		weightbias = []
		
		for i in range(len(self.shape)-1):
			weightbias.append(self.layers[i+1]['weights'])
			weightbias.append(self.layers[i+1]['biases'])
		
		if debug_other:
			print("Variables:")
			for v in weightbias:
				print(v)
				
		with tf.variable_scope('network_{}/init'.format(self.net_index), reuse=tf.AUTO_REUSE):
				
			if not self.init['defined']:
				self.setup_init(weightbias)
				self.train_step_index = 0
				
			self.sess.run([self.init['loss'], self.init['weightbias']])
			if self.optimizer == 'momentum':
				self.sess.run([self.init['momentum']])
			elif self.optimizer == 'adadelta':
				self.sess.run([self.init['adadelta']])
			elif self.optimizer == 'adam':
				self.sess.run([self.init['adam']])
		
	
	def increment_train_step(self):
		train_step_index += 1

	def train(self, train_steps, training_inputs, training_outputs, spam=None, sleep=0):
		showSpam = True
		if spam == 0: # Only show final result
			showSpam = False
		if spam is None:
			spam = train_steps//10
		if spam == 0: # Avoid modulo by zero
			spam = 1
		spc = len(str(train_steps))
		c_net = self
		sess = self.sess
		print("Training network #{} (type={})".format(self.net_index, self.optimizer))

		#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		#run_metadata = tf.RunMetadata()

		for step in range(train_steps):
			c_net.train_step_index += 1
			_, loss, _, s = sess.run([c_net.train_step,
								      c_net.error_tensor,
								      c_net.loss_assign,
								      c_net.summary],
			feed_dict={c_net.inputs: np.array(training_inputs),
			  c_net.desired_outputs: np.array(training_outputs)})
			#options=run_options, run_metadata=run_metadata)
			
			self.writer.add_summary(s, global_step=c_net.train_step_index)
			#self.writer.add_run_metadata(run_metadata, "network_{}_run_{}".format(self.net_index, c_net.train_step_index), global_step=c_net.train_step_index)
			
			if step%spam==0:
				time.sleep(sleep)
				if showSpam:
					#print(self.summary)
					print("    Step {0:>{spc}} loss: {loss}".format(step, loss=loss, spc=spc), end='\n')
		print("Final loss (step {}, total {}): {}".format(train_steps, c_net.train_step_index, loss))
		
	
	#def train(self, train_steps, spam=None, sleep=0):
		#assert current_net in range(1, total_nets+1), "Can't train net {}, does not exist".format(current_net)
		#print("Training: {}".format(current_net))
		#if spam is None:
			#spam = train_steps//10
		#for i in range(train_steps):
			#self.increment_train_step()
			#_, loss, _, s = sess.run([train_step, self.error_tensor, loss_assign, summary],
							#feed_dict={inputs: np.array(training_inputs),
							#desired_outputs: np.array(training_outputs)})
			#t_loss = loss.eval(session=sess)
			
			#if t_loss < 1e9:
				#writer.add_summary(s, global_step=train_step_index)
				
			#if i % spam == 0:
				#time.sleep(sleep)
				#print("Current loss for net {}: {}".format(net_index, t_loss), end="\n")
		#print("Final loss: {}".format(t_loss))
		
	def initialize_search(self, search_string_list, debug_variables=False):
		# Note, this adds stupid extra edges connecting all the networks because it searches all of them
		if type(search_string_list) is str:
			search_string_list = [search_string_list]
		assert type(search_string_list) is list, "input needs to be a string or list of strings"
		variables = {}
		var_inits = []
		global_vars = tf.global_variables()
		for string_index, search_string in enumerate(search_string_list):
			print("Searching for: '{}' in net {}".format(search_string, self.net_index))
			with tf.variable_scope('network_{}/init'.format(self.net_index), reuse=tf.AUTO_REUSE):
				uninitialized = tf.report_uninitialized_variables(name=search_string+'_search')
				variables[string_index] = []
				uninitialized_list = self.sess.run(uninitialized).tolist()
				for i, v in enumerate(global_vars):
					name = bytes(v.name.split(':')[0].encode('ASCII'))
		#         print(name, end='  ')
					if name in uninitialized_list:
						# Check it's in the correct net
						if (bool(re.search(bytes('^network_{}'.format(self.net_index), 'ASCII'), name))):
							# Check the name is correct. Also has a wildcard ending to match suffixes
							if bool(re.search(bytes('{}.*$'.format(search_string), 'ASCII'), name)):
								if debug_variables:
									print("{} matches '{}' pattern".format(name.decode('ASCII'), search_string))
								variables[string_index].append(v)
							else:
								if debug_variables:
									print("{} does not match '{}' pattern".format(name.decode('ASCII'), search_string))
						else:
							if debug_variables:
								print("{} is not in net {}".format(name.decode('ASCII'), net_index))
					else:
						if debug_variables:
							print(name.decode('ASCII') + " is already initialized")
				if debug_variables:
					print("Variables: " + str(variables))
				var_inits.append(tf.variables_initializer(variables[string_index], name=search_string + '_init'))
		return var_inits
