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
import globals

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
	def savHist():
		"""
			saves history to blabla.txt
		"""
		pass


class Uav(abc.ABC):
	"""docstring for uav"""
	def __init__(self, position, color, max_roll):
		self.history = AnimationHistory(color)
		self.position = position
		self.roll = random.randint(-20, 20)
		self.yaw = random.randint(0,360)
		self.speed = 2.5 #constant as the paper stated (m/s)
		self.hist = []
		self.color = color
		self.max_roll = max_roll #degrees

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
		Uav.__init__(self, position, color, max_roll=globals.AI_MAX_ROLL)
		

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
				for _ in range(1):
					self.myApplyAction(myPolicy)
				for _ in range(1):
					rival.myApplyAction(rivalPolicy)

				#print(curPosition,"new: ",self.position," ",myPolicy)
				#print(rivalCurPosition,"Rivalnew: ",rival.position," ",rivalPolicy)



				wB = self.getAng(self.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
				#tB2 = self.getAng(rival.yaw, rival.position[0], rival.position[1], self.position[0], self.position[1])
				tB = 180-self.getAng(rival.yaw, rival.position[0], rival.position[1], self.position[0], self.position[1])
				#print("wb",wB,"tb",tB)
				#time.sleep(0.001)
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
	def __init__(self, position, color,Beta=globals.Beta):
		Uav.__init__(self, position, color, max_roll=globals.REINFORCEMENT_MAX_ROLL)
		self.featureSpace = []
		self.Beta = Beta
		self.currJ = 0
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
		#features = [AA,ATA, R, Sa]
		#features = [AA,ATA, absAA, R, AAp, ATAmin, Sa, Sr, absHCA]
		#features = [absAA, R, AAp, ATAmin, Sa, Sr, absHCA, r_yaw, b_yaw]
		features = [AA,ATA, absAA, R, AAp, ATAmin, Sa, Sr, absHCA]

		for feature in features:
			self.featureSpace.append(feature)

		"""for i in range(len(features)):
			for j in range(i,len(features)):
				self.featureSpace.append(features[i] * features[j])"""
		#print(len(self.featureSpace))
		return self.featureSpace
	def takeActionApprox(self, rival, record = True):
		"""
			Takes Action according to uav's policy
		"""
		bx, by = self.position[0], self.position[1]
		rx, ry = rival.position[0], rival.position[1]

		b_yaw  = self.yaw
		b_roll = self.roll

		r_yaw  = rival.yaw
		r_roll = rival.roll
		actions = ["right", "left", "forward"]
		maxJ = -100

		estimated_rival_action = rival.takeAction(self, False)
		rival.myApplyAction(estimated_rival_action)

		for action in actions:
			self.myApplyAction(action)
			ATA = self.getAng(self.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
			AA = self.getAng(rival.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
			R = ((self.position[0]-rival.position[0])**2 + (self.position[1]-rival.position[1])**2)**0.5

			self.createFeatureSpace(ATA,AA,R, rival.yaw, self.yaw)
			temp = GFunction(ATA, AA, R) + 0.95*self.Beta @ self.featureSpace #0.95 discount factor
			#print(action[0], ":", temp, "ATA", ATA, "AA", AA)
			if ATA<0 or AA<0:
				print("&&&&&&&&&")
				time.sleep(10)
			#time.sleep(0.02)
			if(temp>maxJ):
				maxJ = temp
				optimal_action = action
			self.position[0], self.position[1], self.yaw, self.roll = bx, by, b_yaw, b_roll
		rival.position[0], rival.position[1], rival.yaw, rival.roll = rx, ry, r_yaw, r_roll
		
		#self.hist.append([self.position.copy(), -1, u])
		return optimal_action

	def takeAction(self, rival, record = True):
		actions = ["right", "left", "forward"]
		bx, by = self.position[0], self.position[1]
		rx, ry = rival.position[0], rival.position[1]

		b_yaw  = self.yaw
		b_roll = self.roll

		r_yaw  = rival.yaw
		r_roll = rival.roll
		maxJ = -100
		for ub in actions:
			estimated_rival_action = rival.takeAction(self, False)
			self.myApplyAction(ub)
			rival.myApplyAction(estimated_rival_action)
			for _ in range(globals.Nrools):
				u_nom = self.takeActionApprox(rival)
				estimated_rival_action = rival.takeAction(self, False)
				self.myApplyAction(u_nom)
				rival.myApplyAction(estimated_rival_action)

			ATA = self.getAng(self.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
			AA = self.getAng(rival.yaw, self.position[0], self.position[1], rival.position[0], rival.position[1])
			R = ((self.position[0]-rival.position[0])**2 + (self.position[1]-rival.position[1])**2)**0.5

			self.createFeatureSpace(ATA,AA,R, rival.yaw, self.yaw)
			temp = GFunction(ATA, AA, R) + 0.95*self.Beta @ self.featureSpace #0.95 discount factor
			if(temp > maxJ):
				maxJ = temp
				optimal_action = ub
			self.position[0], self.position[1], self.yaw, self.roll = bx, by, b_yaw, b_roll
			rival.position[0], rival.position[1], rival.yaw, rival.roll = rx, ry, r_yaw, r_roll
		self.currJ = maxJ
		self.history.addToHistory(self.position.copy(), maxJ, optimal_action)

		return optimal_action
class Environment(object):
	"""
		The environment to simulate air combat 
	"""
	def __init__(self,Beta=globals.Beta):
		print("Initiliazing The Environment")
		self.Beta = Beta
		self.dt = 0.05 
		self.roll_speed = 45 #degrees/s
		self.g = 9.8 #yer cekimi
		#self.blue_uav = AI([100,100,100], "blue")
		#self.red_uav = AI([1000,1000,100], "red")
		

	def simulate(self):
		"""
			simulates the air combat for 0.25s
		"""
		ub = ur = ""
		tresh = 3.5
		i = 0
		while(self.blue_uav.currJ < tresh):#while not reinforcement kazanmak
			print("ii:", i)
			i+=1
			if(i>10_000):
				break
			for j in range (5):
					if(j%5 == 0):
						ub = self.blue_uav.takeAction(self.red_uav)
						ur = self.red_uav.takeAction(self.blue_uav, True)

					self.blue_uav.myApplyAction(ub)
					self.red_uav.myApplyAction(ur)
					#time.sleep(dt)

	def simulate_n(self, n):
		for _ in range(n):
			self.blue_uav = ReinforcementModule([random.randint(-5, 5), random.randint(-5, 5), 100], "blue",self.Beta)
			self.red_uav = AI([random.randint(-5, 5), random.randint(-5, 5), 100], "red")
			self.simulate()
			#Thread(target=self.simulate).start()
			show(self)


if __name__=="__main__":
	Beta = globals.Beta
	env = Environment(Beta=Beta)
	env.simulate_n(3)
#show(env)




