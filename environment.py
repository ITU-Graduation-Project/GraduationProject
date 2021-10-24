import numpy as np
import time 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.core.fromnumeric import argmax, argmin
from numpy.lib.function_base import append
from animation import show
import random

import abc
import math

from threading import Thread

class AnimationHistory():
	def __init__(self,color):
		print("history created")
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
		print("worlddd")
		self.history = AnimationHistory(color)
		self.position = position
		self.roll = 0
		self.yaw = 0
		self.speed = 2.5 #constant as the paper stated (m/s)
		self.hist = []
		self.color = color
		if(color == "blue"):
			self.max_roll = 23 #degrees
		elif(color == "red"):
			self.max_roll = 18 #degrees

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
	def takeAction(self, rival):
		pass

class AI(Uav):

	def __init__(self, position, color):
		Uav.__init__(self,position,color)
		print("helloo")

	def findAngle(self,position1,position2):

		vector1 = [0,1]

		vector2 = [abs(position2[1]-position1[1]),abs(position2[0]-position1[0])]

		unit_vector_y = vector1 / np.linalg.norm(vector1)
		unit_vector = vector2 / np.linalg.norm(vector2)

		#print(unit_vector_y,unit_vector)
		dot_product = np.dot(unit_vector_y, unit_vector)
		angle = math.degrees(np.arccos(dot_product))

		return abs(90 - angle)

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
		print(valueTable[i], policies[i])
		print(valueTable)
		print("######")
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

	def takeAction(self, rival):
		"""
			Takes Action according to uav's policy
		"""
		value, u = self.lookAhead(rival)
		
		self.history.addToHistory(self.position.copy(),value,u)
		return u
	
class ReinforcementModule(Uav):

	def takeAction(self, rival):
		"""
			Takes Action according to uav's policy
		"""
		#print("in reinforce")
		#print(self.roll%360,self.yaw%360)
		if (random.randint(0,10) < 1.5):
			u = "forward"
		elif(random.randint(0,10) < 5.5):
			u = "right"
		else:
			u = "left"
		self.history.addToHistory(self.position.copy(), -1, u)
		#self.hist.append([self.position.copy(), -1, u])
		return u



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

		if u != "forward":
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
			print(i)
			for j in range (5):
					if(j%5 == 0):
						ub = self.blue_uav.takeAction(self.red_uav)
						ur = self.red_uav.takeAction(self.blue_uav)

					self.blue_uav = self.applyAction(self.blue_uav, ub)
					self.red_uav = self.applyAction(self.red_uav, ur)
					#time.sleep(dt)

	def simulate_n(self, n):
		for _ in range(n):
			self.blue_uav = AI([random.randint(500, 1500), random.randint(500, 1500), 100], "blue")
			self.red_uav = ReinforcementModule([random.randint(500, 1500), random.randint(500, 1500), 100], "red")
			self.simulate()
			#Thread(target=self.simulate).start()
			show(env)



env = Environment()

env.simulate_n(2)
#env.show()
show(env)