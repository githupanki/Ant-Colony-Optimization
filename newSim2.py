# AA 222 Final Project Code
# Jeremy Morton and Mark Paluta -- 5/29/15

from devices.device import Device
import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle
import pickle
import pylab

''' One type of device, described below:
==================================================================================
To create a device:
device = Device(powerLevel, duration, step)

powerLevel 	-- power consumed by device
duration 	-- length of time (in hours) that device draws power
step 		-- time step for simulation, in minutes (default value is 60)

Other Parameters:
start 		-- time step at which device begins using power, initialized to None
feasible 	-- list of time steps in which device could begin using power
pred 		-- device that precedes this device in priority
succ 		-- device that succeeds this device in priority
LS 			-- latest possible time at which device could begin consuming power
ES 			-- earliest possible time at which device could begin consuming power
LF 			-- latest possible time at which device could finish consuming power
EF 			-- earliest possible time at which device could finish consuming power
start 		-- time at which device starts consuming power
index		-- index of devices within device list


The methods for a Type 1 device are:
setStart(time) -- sets the time at which a device begins using power.
	time is in units of time steps
status(time) -- returns a boolean for the provided time step to indicate
	whether the device is drawing power at the given time
'''

# Function to find baseline power usage.  Takes in an average power consumption value, amplitude, 
# and time step size, and returns a list containing the power value at each time step.  The values
# are based on a sine curve centered at 10:30am
def getBaseline(avg, amp, step=60, simTime=24):
	values = np.zeros(simTime*60/step)

	# Create function to calculate power usage 
	f = lambda x: amp*np.sin(2*np.pi/24*x +7.5/24*2*np.pi) + avg
	#f = lambda x: min(amp*np.sin(2*np.pi/24*x +7.5/24*2*np.pi) + avg, avg )
	for i in range(simTime*60/step):
		values[i] = f(i*step/60.)
	return values

def getMaximum(avg, amp, step=60, simTime=24):
	values = np.zeros(simTime*60/step)

	# Create function to calculate power usage 
	f = lambda x: max(amp*np.sin(2*np.pi/24*x +12.0/24*2*np.pi) + avg, avg-0.35*amp)
	for i in range(simTime*60/step):
		values[i] = f(i*step/60.)
	return values

# This function takes in a dictionary of values and probabilities, and chooses one value based
# on the probabilities
def weighted_choice(choices):

	# In case probabiliities don't sum to one
	total = sum(w for w in choices.values())
	r = random.uniform(0, total)
	upto = 0
	for c in choices.keys():
		if upto + choices[c] > r:
			return c
		upto += choices[c]


# Function takes in a number n and time step size and generates n Type 1 devices.
def generateDevice(n, step=60):

	# Calculate # of time steps in an hour
	factor = 60/step

	# List of possible duration times
	duration = {1: .35, 2: .25, 3: .2, 4:.1, 5:.1}
	#duration = {1: .5, 5:.5}

	# If steps are not an hour
	if step != 60:

		# Initialize new dictionary
		newDur = {}

		# Create new dictionary for possible task durations
		for key in duration.keys():
			for i in range(factor):
				newDur[factor*key + i] = duration[key]

		# Define new dictionaries
		duration = newDur

	# Initialize list of devices
	devices = []
	for i in range(n):
		# Choose duration
		dur = weighted_choice(duration)

		# Choose power
		power = np.random.uniform(1e3, 10e3)
		# 50% chance to reduce device power by one order of magnitude
		# if np.random.randint(0, 10) != 0:
		# 	power *= 0.1

		# Create device
		device = Device(power, dur, step)

		devices.append(device)
		device.index = len(devices) - 1

	return devices

# This takes in a list of devices and finds the predecessor and successor
# for each device.  If a device is the first device in a household, its
# predecessor is start.  If it is the last device in a household, its successor
# is finish
def predSucc(devices, start, finish, chain=5):
	for i in range(len(devices)):
		# If first device in household
		if i % chain == 0:
			devices[i].pred = start
		else:
			devices[i].pred = devices[i - 1]

		# Last device in household
		if i % chain == chain - 1:
			devices[i].succ = finish
		else:
			devices[i].succ = devices[i + 1]

# This method takes in a device and finish time for the simulation and finds
# the latest possible finish and start time for each device
def getLate(device, T):
	# Base case -- last device from household, latest finish time is T,
	# latest start time is T minus duration
	if device.succ.duration == 0:
		device.LF = T
		device.LS = device.LF - device.duration
	
	# Recursive case -- determine late finish time based on late start time
	# of successor device
	else:
		device.LF = getLate(device.succ, T)
		device.LS = device.LF - device.duration
	return device.LS

# This method takes in a device and finds the earliest possible start and
# finish times for the device and its successors.
def getEarly(device, chain=5):
	for i in range(chain):
		if i == 0:
			device.ES = 0
		else:
			device.ES = device.pred.EF
		device.EF = device.ES + device.duration

		# Move on to next device
		device = device.succ


# This method takes in a list of devices and initializes all of them
def initialize(devices, chain=5):

	# Define devices to represent start and finish of simulation
	start = Device(0, 0)
	start.start = 0

	start.finish = 0
	finish = Device(0, 0)

	# Set length of simulation
	T = 23

	# Find predecessor and successor for each device
	predSucc(devices, start, finish, chain)

	# Call getEarly and getLate for every five devices
	for i in range(0, len(devices), chain):
		getEarly(devices[i], chain)
		getLate(devices[i], T)

# This method takes in devices, loops over them, and returns a list of
# all devices that are eligible at a given time step
def getEligible(devices, time, power, maximum):
	eligible = []
	for i in range(len(devices)):

		# If device has not started drawing power but its predecessor has
		if devices[i].start == None and devices[i].pred.start != None:

			# Check that predecessor has finished and load threshold won't 
			# be exceeded if device begins drawing power
			if time >= devices[i].pred.finish and\
			 all([power[t] + devices[i].powerLevel <= maximum[t] for t in range(time, time + devices[i].duration)]):
				eligible.append(i)
	return eligible

# This heuristic works to schedule activities according to growing
# values of latest finish time
def latestFinish(devices, index, eligible):
	maxLF = max([devices[i].LF for i in eligible])
	return maxLF - devices[index].LF + 1

# This heuristic works to schedule activities according to growing
# values of latest start time
def latestStart(devices, index, eligible):
	maxLS = max([devices[i].LS for i in eligible])
	return maxLS - devices[index].LS + 1

# This heuristic tends to prioritize devices with the smallest difference
# between their latest start time and earliest start time
def minSlack(devices, index, eligible):
	maxSlack = max([devices[i].LS - devices[i].ES for i in eligible])
	return maxSlack - (devices[index].LS - devices[index].ES) + 1

# This heuristic prioritizes activities with the longest cumulative duration
# between them and their successors
def GRPWA(devices, index, eligible):
	minDur = min([totalDur(devices[i]) for i in eligible])
	return totalDur(devices[index]) - minDur + 1

# This method takes in a device and returns the sum of the device's duration
# and that of all of its successors
def totalDur(device):
	total = 0
	while device.duration != 0:
		total += device.duration
		device = device.succ
	return total

def getEta(devices, index, eligible, method):
	if method == 'latestFinish':
		return latestFinish(devices, index, eligible)
	elif method == 'latestStart':
		return latestStart(devices, index, eligible)
	elif method == 'minSlack':
		return minSlack(devices, index, eligible)
	elif method == 'GRPWA':
		return GRPWA(devices, index, eligible)

# This method performs a pheromone update.  It first decreases the pheromone
# values thoughout the matrix, the strengthens the values that correspond
# to assignments to lead to local and global best solutions
def pherUpdate(tau, rho, tGlobal, tLocal, globalDict, localDict):
	tau = np.multiply(tau, (1 - rho))
	for col in range(tau.shape[1]):
		tau[globalDict[col]][col] += rho/2/tGlobal
		tau[localDict[col]][col] += rho/2/tLocal

# This method calculates the probability that any eligible device is chosen
# at a given time step, based on pheromones and heuristic information
def getProbs(devices, time, tau, alpha, beta, eligible):
	probabilities = {}
	for i in eligible:
		eta = getEta(devices, i, eligible, "latestFinish")
		product = tau[time][i]**alpha * eta**beta
		probabilities[devices[i]] = product
	total = sum(probabilities.values())
	# Return normalized values, which correspond to probabilities
	return {k : probabilities[k]/total for k in probabilities.keys()}

# This method calculates the probability that any eligible device is chosen
# at a given time step, based on (summation) pheromones and heuristic information
def getProbsSum(devices, time, tau, alpha, beta, gamma, eligible):
	probabilities = {}
	for i in eligible:
		eta = getEta(devices, i, eligible, "GRPWA")

		# Calculate sum of gamma times pheromones for all
		# time steps up to current
		tauSum = gamma * tau[0][i]
		if time > 0:
			for t in range(1, time):
				tauSum += gamma**(time - t) * tau[t][i]

		product = tauSum**alpha * eta**beta
		probabilities[devices[i]] = product
	total = sum(probabilities.values())

	# Return normalized values, which correspond to probabilities
	return {k : probabilities[k]/total for k in probabilities.keys()}

# This method takes in a device and power curve, and adds the devices usage
# to the power curve
def updatePower(device, power, simTime=24):
	for time in range(device.start, min(device.finish, simTime*60/step)):
		power[time] += device.powerLevel

# This method takes in a list of devices and returns the makespan, which is the difference between
# the earliest device start time and the latest device finish time
def findMakeSpan(devices, chain=5):
	minimum = min(device.start for device in devices)

	maximum = max(device.finish for device in devices)
	return maximum - minimum

# This mthod resets the start and finish times for all devices
def resetDevices(devices):
	for device in devices:
		device.start = None
		device.finish = None

def deviceQueue(devices, baseline, simTime, maximum):
	power = list(baseline)
	# Loop over times in day
	for time in range(simTime*60/step):
		# Find eligible devices at time step
		eligible = getEligible(devices, time, power, maximum)
		# While eligible devices exist and power threshold has not been exceeded
		while len(eligible) > 0:

			# Select a device and define start and finish time
			chosen = devices[eligible[0]]
			chosen.start = time
			chosen.finish = time + chosen.duration
			for steps in range(chosen.finish, 24*60/step):
				qAll[steps] += 1

			# Update power usage
			updatePower(chosen, power, simTime)

			# Remove device from list of eligible devices
			eligible = getEligible(devices, time, power, maximum)

	# Plot power
	plt.step(a, [power[i]/1000 for i in range(len(a))], linewidth=2.0, where ='post')
	legendvector.append('Descending Chain Duration')
	# Calculate makespan
	return (findMakeSpan(devices, chain), power)

def antColonyOptimize(n_iter, alpha, beta, gamma, rho, gmax, n, step, n_ants, simTime, chain, base, baseline, maximum, legendvector, tau, tGlobal, tLocal, globalDict, localDict, power, allocation):

	# Loop for desired iterations
	for i in range(n_iter):
		# Loop over each ant
		for j in range(n_ants):

			# Loop over times in day
			for time in range(simTime*60/step):

				# Find eligible devices at time step
				eligible = getEligible(devices, time, power, maximum)

				# While eligible devices exist and power threshold has not been exceeded
				while len(eligible) > 0:

					# Find probability of choosing each device
					probs = getProbsSum(devices, time, tau, alpha, beta, gamma, eligible)

					# # Find probability of choosing each device
					# probs = getProbs(devices, time, tau, alpha, beta, eligible)

					# Select a device and define start and finish time
					chosen = weighted_choice(probs)
					chosen.start = time
					chosen.finish = time + chosen.duration
					for steps in range(chosen.finish, 24*60/step):
						allocation[steps] += 1

					# Update power usage
					updatePower(chosen, power, simTime)

					# Update list of eligible devices
					eligible = getEligible(devices, time, power, maximum)

			
			# Calculate makespan
			t = findMakeSpan(devices, chain)

			# See if makespan is local or global best and store if it is
			if t < tLocal:
				tLocal = t
				localDict = {i : devices[i].start for i in range(len(devices))}

				if t < tGlobal:
					tGlobal = t
					globalDict = localDict
					bestPower = list(power)
					bestAllocation = allocation
					bestTimes.append(t)
			allocation = {k : 0 for k in range(simTime*60/step)}

		# Update pheromones
		pherUpdate(tau, rho, tGlobal, tLocal, globalDict, localDict)

		if i % 5 == 0:
			tau = np.ones((simTime*60/step, n))

		# Reset power and devices	
		power = list(baseline)
		resetDevices(devices)

	plt.step(a, [bestPower[i]/1000 for i in range(len(a))], linewidth=2.0, where = 'post')
	legendvector.append('Ant Colony')

	print bestTimes
	return (bestAllocation, bestPower, tGlobal)

def immediatePlacement(devices, n):

	for i in range(n):

		chosen = devices[i]
		chosen.start = np.random.randint(0, 10)

		chosen.finish = chosen.start + chosen.duration
		for steps in range(chosen.finish, 24*60/step):
			allocation[steps] += 1

		# Update power usage
		updatePower(chosen, power, simTime)

	bestPower = list(power)
	plt.plot(a, [bestPower[i]/1000 for i in range(len(a))], linewidth=2.0)
	legendvector.append('Immediate Placement')

# Define simulation parameters
n_iter = 500
alpha = 1
beta = 2#np.linspace(2, 0, n_iter)
gamma = 1
rho =  5e-3
gmax = 10
n = 50
step = 10
n_ants = 5
simTime = 48
chain = 1
tGlobal = 1e6
tLocal = 1e6
improvement = []

# Define mean baseline power level
base = 2.15e5

# Find baseline power usage
baseline = getBaseline(base, .3*base, step, simTime)

# Set maximum power usage
maximum = getMaximum(3.75e5, 40000, step, simTime)
#maximum = [baseline[i]+3e4 for i in range(len(baseline))]
#maximum = [3.5e5 for i in range(len(baseline))]

# Create plot
a = range(simTime*60/step)
legendvector = []

n_sim = 1
for i in range(n_sim):
	#Create and initialize devices
	devices = generateDevice(n, step)
	initialize(devices, chain)

	# # Write devices to file
	# file_Name = "deviceFile"
	# fileObject = open(file_Name,'wb') 
	# pickle.dump(devices,fileObject)   
	# fileObject.close()

	# # Download devices from file
	# fileObject = open("deviceFile",'r') 
	# devices = pickle.load(fileObject)   
	# fileObject.close()

	# Parameters that must reset before for each simulation
	globalDict = {}
	localDict = {}
	power = list(baseline)
	bestPower = list(power)
	bestTimes = []
	allocation = {k : 0 for k in range(simTime*60/step)}
	qAll = {k : 0 for k in range(simTime*60/step)}
	bestAllocation = {}

	# Initialize pheromone matrix
	#tau = np.ones((simTime*60/step, n))

	#(bestAllocation, bestPower, t_ACO) = antColonyOptimize(n_iter, alpha, beta, gamma, rho, gmax, n, step, n_ants, simTime, chain, base, baseline, maximum, legendvector, tau, tGlobal, tLocal, globalDict, localDict, power, allocation)

	#for i in range(len(devices)):
	#	devices[i].totalDur = GRPWA(devices, i, range(len(devices)))

	#devices.sort(key=lambda x: x.totalDur, reverse=True)
	#devices.sort(key=lambda x: x.duration, reverse=True)

	#(t_queue, qPower) =  deviceQueue(devices, baseline, simTime, maximum)
	#print t_queue

	#improvement.append((t_queue - t_ACO)*step)

	immediatePlacement(devices, n)

#print np.mean(improvement)

plt.plot(a, [baseline[i]/1000 for i in range(len(a))], a, [maximum[i]/1000 for i in range(len(a))], linewidth = 2.0)
legendvector.append('Baseline')
legendvector.append('Maximum')

plt.legend(legendvector, loc = 'best')
plt.xlim([0, 24*60/step])
avals = [a[i] for i in range(0, 24*60/step, 60/step*2)]
plt.xticks(avals, [18, 20, 22, 0, 2, 4, 6, 8, 10, 12, 14, 16])
plt.xlabel('Time (hour)', fontsize=16)
plt.ylabel('Power (kW)', fontsize=16)
plt.grid()
plt.savefig('figure1.pdf', bbox_inches='tight')
plt.show()

# Right now these second and third plots are just using the final simulation's values.

# plt.figure(2)
# plt.plot(a, bestAllocation.values(), a, qAll.values(), linewidth = 2)
# plt.legend(['Ant Colony', 'Queue'], loc='best')
# plt.xlim([0, 24*60/step])
# plt.xticks(avals, [18, 20, 22, 0, 2, 4, 6, 8, 10, 12, 14, 16])
# plt.xlabel('Time (hour)', fontsize=16)
# plt.ylabel('Devices Allocated', fontsize=16)
# plt.grid()

# plt.show()

# devicePower = [(bestPower[i] - baseline[i])/1000 for i in range(len(baseline))]
# qDevicePower = [(qPower[i] - baseline[i])/1000 for i in range(len(baseline))]

# plt.figure(3)
# plt.plot(a, devicePower, a, qDevicePower, linewidth = 2)
# plt.legend(['Ant Colony', 'Queue'], loc='best')
# plt.xlim([0, 24*60/step])
# plt.xticks(avals, [18, 20, 22, 0, 2, 4, 6, 8, 10, 12, 14, 16])
# plt.xlabel('Time (hour)', fontsize=16)
# plt.ylabel('Power Consumed by Devices (kW)', fontsize=16)
# plt.grid()
# plt.show()










































