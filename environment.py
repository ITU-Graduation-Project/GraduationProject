import numpy as np
import time 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.core.fromnumeric import argmax, argmin
from numpy.lib.function_base import append
from animation import show
import random
from reward import GFunction
import abc
import math

from threading import Thread

class AnimationHistory():
	def __init__(self,color):
		self.color = color
		self.position_history = []
		self.value_history = []
	
	def addToHistory(self,position,value,u):
		self.position_history.append(position)
		self.value_history.append([value, u])
	
	def getHistory(self):
		position_history = self.position_history
		position_history = np.array(position_history)[:,0:2]

		value_history = self.value_history
		value_history = np.array(value_history,dtype=object)

		print(position_history.shape)
		return position_history, value_history


class Uav(abc.ABC):
	"""docstring for uav"""
	def __init__(self, position, color):
		self.history = AnimationHistory(color)
		self.position = position
		self.roll = random.randint(-20,20)
		self.yaw = random.randint(0,360)
		self.speed = 2.5 #constant as the paper stated (m/s)
		self.hist = []
		self.color = color
		if(color == "blue"):
			self.max_roll = 23 #degrees
		elif(color == "red"):
			self.max_roll = 18 #degrees

	def getAng(self, heading, Bx, By, Rx, Ry):
		temp = math.sqrt((Ry-By)**2 + (Rx-Bx)**2)
		LosX = (Rx-Bx)/temp
		LosY = (Ry-By)/temp
		BlueVecY = math.cos(heading*math.pi/180)
		BlueVecX = math.sin(heading*math.pi/180)

		vector_1 = [BlueVecX, BlueVecY]
		vector_2 = [LosX, LosY]

		unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
		unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product)

		#print("angle",angle*180/math.pi)
		return angle*180/math.pi


	def myApplyAction(self, u):
		if u == "left":
			self.roll = self.roll - 45 * 0.05
			if self.roll < -self.max_roll:
				self.roll = -self.max_roll
		elif u == "right":
			self.roll = self.roll + 45 * 0.05
			if self.roll > self.max_roll:
				self.roll = self.max_roll

		yaw_diff = (9.8/self.speed)*np.tan(np.radians(self.roll))
		self.yaw = self.yaw + yaw_diff * 0.05

		self.yaw = self.yaw % 360
		
		self.position[0] = self.position[0] + 0.05*np.sin(np.radians(self.yaw)) # x = x + sin(yaw)*dt
		self.position[1] = self.position[1] + 0.05*np.cos(np.radians(self.yaw)) # y = y + cos(yaw)*dt
		#print(self.position)


	@abc.abstractmethod
	def takeAction(self, rival, record = True):
		pass

class AI(Uav):

	def __init__(self, position, color):
		Uav.__init__(self,position,color)

	def lookAhead(self, rival):
		
		curPosition = self.position.copy()
		curRoll = self.roll
		curYaw = self.yaw

		rivalCurPosition = rival.position.copy()
		rivalCurRoll = rival.roll
		rivalCurYaw = rival.yaw

		policies = ["left", "right", "forward"]
		valueTable = []
		for myPolicy in policies:
			policyValues = []
			for rivalPolicy in policies:
				self.position = curPosition.copy()
				self.roll = curRoll
				self.yaw = curYaw

				rival.position = rivalCurPosition.copy()
				rival.roll = rivalCurRoll
				rival.yaw = rivalCurYaw
				for _ in range(5):
					self.myApplyAction(myPolicy)
				for _ in range(5):
					rival.myApplyAction(rivalPolicy)

				#print(curPosition,"new: ",self.position," ",myPolicy)
				#print(rivalCurPosition,"Rivalnew: ",rival.position," ",rivalPolicy)



				wB = self.getAng(self.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
				tB = self.getAng(rival.yaw, rival.position[0], rival.position[1], self.position[0], self.position[1])
				Sa = 1 - (wB + tB) / 180

				#print("Sa value: ", Sa,"wb: ", wB, "tb: ", tB , "self yaw ,rival yaw: ",self.yaw," ",rival.yaw)
				policyValues.append(Sa)
				#time.sleep(5)
			minPolicy = argmin(policyValues)
			#print("$$$$$$$$$$$$$$$$$$$$$ ",minPolicy,policyValues[minPolicy])
			valueTable.append(policyValues[minPolicy])

		i = argmax(valueTable)
		#print(valueTable[i], policies[i])
		#print(valueTable)
		#print("######")
		#time.sleep(15)

		self.position = curPosition.copy()
		self.roll = curRoll
		self.yaw = curYaw

		rival.position = rivalCurPosition.copy()
		rival.roll = rivalCurRoll
		rival.yaw = rivalCurYaw

		if valueTable[i]<=-1:
			print("$$$$$$$$$$$$$$$$$$$$$$$$",valueTable[i])
			print(valueTable)
			#time.sleep(3)

		return valueTable[i], policies[i]

	def takeAction(self, rival, record):
		"""
			Takes Action according to uav's policy
		"""
		value, u = self.lookAhead(rival)
		if(record):
			self.history.addToHistory(self.position.copy(),value,u)
		return u
	
class ReinforcementModule(Uav):
	def __init__(self, position, color):
		Uav.__init__(self,position,color)
		self.featureSpace = []

	def createFeatureSpace(self, ATA, AA, R, r_yaw, b_yaw, Rd = 3, k = 0.1):
		absAA = abs(AA)
		R = R 
		AAp = max(0,AA) 
		ATAmin = min(0,ATA) 
		Sa = 1 - ((1-AA/180)+(1-ATA/180))
		Sr  = np.exp(-abs(R-Rd)/(180*k))
		absHCA = abs(AA - ATA) 
		r_yaw = r_yaw 
		b_yaw = b_yaw 
		self.featureSpace = []
		features = [absAA, R, AAp, ATAmin, Sa, Sr, absHCA, r_yaw, b_yaw]
		for i in range(len(features)):
			for j in range(len(features)):
				self.featureSpace.append(features[i] * features[j])
		return self.featureSpace
	def takeAction(self, rival, record = True):
		"""
			Takes Action according to uav's policy
		"""
		Beta = np.array([-2.05801371e-07, -1.32105191e-07, -1.32194339e-07, -3.88529437e-08,  3.71688173e-05,  1.20288655e-04,  1.07846023e-07, -5.12031497e-09,  2.63726450e-09,  2.70119213e-07, -2.90321969e-07,  2.71506918e-07, -8.87506979e-10, -3.09720399e-05,  5.71942941e-03,  8.06490763e-08,  8.25889708e-08,  8.42943642e-08, -2.39976770e-07,  2.74210113e-07, -2.39976875e-07, -1.01324281e-14,  3.71520651e-05,  1.20292578e-04,  9.04443683e-08, -2.01610342e-09, -4.89389515e-09,  1.55106906e-17,  6.32357890e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.71520651e-05, -3.09712497e-05,  3.71520651e-05,  0.00000000e+00, -9.89501414e-03, -2.71536528e-01, -1.76638808e-05,  4.95700874e-07,  4.99751008e-07,  1.20292578e-04,  5.71943186e-03,  1.20292578e-04,  0.00000000e+00, -2.71536528e-01,  4.42319430e-01,  8.16284596e-05,  5.76190220e-05,  5.92956581e-05,  9.04443686e-08,  8.06496068e-08,  9.04443686e-08,  0.00000000e+00, -1.76638808e-05,  8.16284596e-05, -8.70711563e-08, -4.35851002e-08, -4.04178984e-08, -2.01610341e-09,  8.25889998e-08, -2.01610341e-09,  0.00000000e+00,  4.95700874e-07,  5.76190220e-05, -4.35851002e-08, -6.04332900e-08, -5.07715534e-08, -4.89389508e-09,  8.42944375e-08, -4.89389508e-09,  0.00000000e+00,  4.99751008e-07,  5.92956581e-05, -4.07012903e-08,  5.21011529e-08, -6.28367432e-08])
		#print("in reinforce")
		#print(self.roll%360,self.yaw%360)
		
		

		bx, by = self.position[0], self.position[1]
		rx, ry = rival.position[0], rival.position[1]

		b_yaw  = self.yaw
		b_roll = self.roll

		r_yaw  = rival.yaw
		r_roll = rival.roll

		estimated_rival_action = rival.takeAction(self, False)
		rival.myApplyAction(estimated_rival_action)

		actions = ["right", "left", "forward"]
		maxJ = -100
		print("*"*10)
		
		for action in actions:
			self.myApplyAction(action)
			ATA = self.getAng(self.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
			AA = 180-self.getAng(rival.yaw, rival.position[0], rival.position[1], self.position[0], self.position[1])
			R = ((self.position[0]-rival.position[0])**2 + (self.position[1]-rival.position[1])**2)**0.5
			self.createFeatureSpace(ATA,AA,R, rival.yaw, self.yaw)
			temp = GFunction(ATA, AA, R) + 0.8*Beta @ self.featureSpace #0.8 discount factor
			print(action[0], ":", temp)
			if(temp>maxJ):
				maxJ = temp
				optimal_action = action
			self.position[0], self.position[1], self.yaw, self.roll = bx, by, b_yaw, b_roll
		rival.position[0], rival.position[1], rival.yaw, rival.roll = rx, ry, r_yaw, r_roll

		self.history.addToHistory(self.position.copy(), -1, optimal_action)
		#self.hist.append([self.position.copy(), -1, u])
		return optimal_action


class Environment(object):
	"""
		The environment to simulate air combat 
	"""
	def __init__(self):
		print("Initiliazing The Environment")
		self.dt = 0.05 
		self.roll_speed = 45 #degrees/s
		self.g = 9.8 #yer cekimi
		self.blue_uav = AI([100,100,100], "blue")
		self.red_uav = ReinforcementModule([1000,1000,100], "red")
		
	
	def applyAction(self, uav, u):
		"""
			applies the given action to given uav
		"""
		

		if u == "left":
			uav.roll = uav.roll - self.roll_speed * self.dt
			if uav.roll < -uav.max_roll:
				uav.roll = -uav.max_roll
		elif u == "right":
			uav.roll = uav.roll + self.roll_speed * self.dt
			if uav.roll > uav.max_roll:
				uav.roll = uav.max_roll

		yaw_diff = (self.g/uav.speed)*np.tan(np.radians(uav.roll))
		uav.yaw = uav.yaw + yaw_diff * self.dt

		uav.yaw = uav.yaw % 360
		
		uav.position[0] = uav.position[0] + self.dt*np.sin(np.radians(uav.yaw)) # x = x + sin(yaw)*dt
		uav.position[1] = uav.position[1] + self.dt*np.cos(np.radians(uav.yaw)) # y = y + cos(yaw)*dt

		return uav

	def simulate(self):
		"""
			simulates the air combat for 0.25s
		"""
		ub = ur = ""
		for i in range(10000):
			for j in range (5):
					if(j%5 == 0):
						ub = self.blue_uav.takeAction(self.red_uav)
						ur = self.red_uav.takeAction(self.blue_uav, True)

					self.blue_uav = self.applyAction(self.blue_uav, ub)
					self.red_uav = self.applyAction(self.red_uav, ur)
					#time.sleep(dt)

	def simulate_n(self, n):
		for _ in range(n):
			self.blue_uav = ReinforcementModule([random.randint(-100, 100), random.randint(-100, 100), 100], "blue")
			self.red_uav = AI([random.randint(-100, 100), random.randint(-100, 100), 100], "red")
			self.simulate()
			#Thread(target=self.simulate).start()
			show(env)



env = Environment()
env.simulate_n(2)
show(env)




