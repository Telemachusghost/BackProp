"""
A module to enable simple back propagation
***WARNING THIS CODE IS MESSY***
Very focused on just getting this to work.
I need to add some type of layer abstraction to make it easier to use.
The meat of the neuron could could use some tlc also.
I have found the best luck using cos() as the activation function 
it converging every time I tried it in less than 100 epochs

Derick Falk
5/31/2018
"""
import random as rd
from math import *



class neuron():
	# Takes parameters for number of weights, activation callback function, learning rate, and whether
	# The node is hidden or an output
	def __init__(self,wn,activation,alpha,t):
		self.pattern = []
		self.hidden=True
		self.output=False
		self.activation = activation
		self.connectionsto=[]
		self.connectionsfrom=[]
		if t == 'hidden':
			self.hidden = True
			self.output = False
			self.err = {}
		elif t=='output':
			self.output = True
			self.hidden= False
			self.err = 0
			
		self.weights = []
		self.previous = []
		self.recv = {}
		for i in range(wn):
			self.weights.append(rd.uniform(-0.5,0.5))
		self.alpha = alpha
		

	# Connects neurons.
	# This could use some work I ended up incrementing a line variable so that the neurons would keep
	# Track of which weight to use when updating the error, basically how the neuron is connected to
	# the next neuron.. Maybe use a linked list
	def connect(self,neuron):
		self.connectionsto.append(neuron)
		neuron.connectionsfrom.append(self)
		

	# Takes the derivative
	# Just a simple function to take derivative not much going on
	def diff(self,function,x):
		delta = 0.01
		xh = delta+x
		num = function(xh) - function(x)
		# print(f"diff at {x}:  {num/delta}")
		return num/delta


	# Propagates a pattern and if a hidden node sends its ouput up to the next layer
	def prop(self,pattern = (0,5)):
		if self.hidden:
			self.y = 0
			self.y_in = 0
			for i in range(len(pattern)):
				 self.y_in += self.weights[i]*pattern[i]
			self.y_in += self.weights[-1]
			
			self.y = self.activation(self.y_in)
			for i in self.connectionsto:
				i.pattern.append(self.y)
		if self.output:
			self.y = 0
			self.y_in = 0
			for i in range(len(self.pattern)):
				 self.y_in += self.weights[i]*self.pattern[i]
			self.y_in += self.weights[-1]
			self.y = self.activation(self.y_in)
			#print(self.y)
			self.pattern = []
		

	# Computes the error for backprop
	# Calculates 
	def error(self,target,inp):
		if self.output:
			self.weight_delta = []
			self.err = (target - self.y)*self.diff(self.activation,self.y_in)
			for i in self.connectionsfrom:
				self.weight_delta.append(self.err*self.alpha*i.y)
			self.weight_delta.append(self.err*self.alpha)
			#print(len(self.connectionsfrom))
				
		elif self.hidden:
			self.weight_delta = []
			error_in = 0
			for i in self.connectionsto:
				error_in += i.err*i.weights[i.connectionsfrom.index(self)]

			error = error_in*self.diff(self.activation,self.y_in)
			# print(error)
			for i in range(len(inp)):
				self.weight_delta.append(self.alpha*error*inp[i])
			self.weight_delta.append(self.alpha*error)
			
	
	# Updates the weights
	def update(self):
		for i in range(len(self.weight_delta)):
			self.weights[i] += self.weight_delta[i]
		# self.weights[-1] += self.weight_delta[-1]
	


# XOR example
def main():
	xordataset = {(1,1):-1, (1,-1):1, (-1,1):1, (-1,-1):-1}
	epochs = 1000
	learning_rate = 0.02
	activationh = lambda x: max(0,x)
	activationo = lambda x: 2/(1+exp(-x))-1
	activationc = lambda x: cos(x)
	activations = lambda x: sin(x)
	
	# Hidden Layer really need to add a layer construct!
	z_1 = neuron(3,activationc,learning_rate,'hidden')
	z_2 = neuron(3,activationc,learning_rate,'hidden')
	# z_3 = neuron(3,activationc,learning_rate,'hidden')
	# z_4 = neuron(3,activationc,learning_rate,'hidden')

	# z_5 = neuron(3,activationc,learning_rate,'hidden')
	# z_6 = neuron(3,activationc,learning_rate,'hidden')
	# z_7 = neuron(3,activationc,learning_rate,'hidden')
	# z_8 = neuron(3,activationc,learning_rate,'hidden')

	# Output layer and connections
	y_1 = neuron(3,activationc,learning_rate,'output')
	z_1.connect(y_1)
	z_2.connect(y_1)
	# z_3.connect(y_1)
	# z_4.connect(y_1)

	# z_5.connect(y_1)
	# z_6.connect(y_1)
	# z_7.connect(y_1)
	# z_8.connect(y_1)

	acc = 0
	total = 0
	dev = 0
	for i in range(epochs):
		for i in xordataset:
			z_1.prop(i)
			z_2.prop(i)

			# z_3.prop(i)
			# z_4.prop(i)
			
			# z_5.prop(i)
			# z_6.prop(i)
			# z_7.prop(i)
			# z_8.prop(i)
			
			y_1.prop()
			y_1.error(target=xordataset[i],inp=i)
			
			z_1.error(target=0,inp=i)
			z_2.error(target=0,inp=i)
			# z_3.error(i)
			# z_4.error(i)

			# z_5.error(i)
			# z_6.error(i)
			# z_7.error(i)
			# z_8.error(i)
			
			y_1.update()
			z_1.update()
			z_2.update()
			# z_3.update()
			# z_4.update()
			# z_5.update()
			# z_6.update()
			# z_7.update()
			# z_8.update()
			# print(z_1.weights)
			dev += (xordataset[i]-y_1.y)**2
			if y_1.y > 0 and xordataset[i] > 0: 
				acc+=1
				print("CORRECT")
			elif y_1.y < 0 and xordataset[i] < 0: 
				acc+=1
				print("CORRECT")
			else:
				print("INCORRECT")
			total += 1
		mse = dev/4
		converged = False
		if mse < 0.05:
			print(f"CONVERGED WITH MSE OF {mse}")
			converged = True
			break
		else:
			dev = 0
		print(f"MSE:{mse}")
	#print(f"****Final Weights****\nz1_weights: {z_1.weights} z2_weights {z_2.weights} y1_weights {y_1.weights}")
	print(f"accuracy: {acc/total}")
	print(f" {total/4} epochs")
	print(f"The seperating line for z1: {(-z_1.weights[0])/z_1.weights[1]}x + {-z_1.weights[2]/z_1.weights[1]}")
	print(f"The seperating line for z2: {(-z_2.weights[0])/z_2.weights[1]}x + {-z_2.weights[2]/z_2.weights[1]}")
	return converged
	#for i in xordataset:



if __name__=="__main__":
	# total = 0
	# for i in range(10):
	t = main()
	# 	if t == True:
	# 		total += 1
	# print(f"converged {total}  times")
