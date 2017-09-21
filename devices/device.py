import numpy as np
import random

# Define class for type 1 devices
class Device:

	# Initialize class object with power level, duration of required power usage, time at which
	# task was assigned, window over which task must be completed, and time step for analysis
	def __init__(self, powerLevel, duration, step = 60):
		self.powerLevel = powerLevel
		self.duration = duration  	# In units of time steps
		self.step = step
		self.pred = None
		self.succ = None
		self.LS = None
		self.ES = None
		self.LF = None
		self.EF = None
		self.start = None
		self.finish = None
		self.index = None

	# Method to set start time for when object will begin using power
	def setStart(self, time):
		self.start = time

	# Returns true if device is currently using power
	def status(self, time):
		return time > self.start and time < self.start + self.duration








