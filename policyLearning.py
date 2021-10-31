from environment import ReinforcementModule, AI, Environment
import numpy as np 
import random
import math
import time
from reward import GFunction
import globals

def calculateS(AA, ATA, R, Rd, k = 0.1):
	ft = 1-(AA/180)
	st = 1-(ATA/180)
	ft = (ft + st)/2
	factor  = np.exp(-abs(R-Rd)/(180*k))
	return ft*factor
def getAng(heading, Bx, By, Rx, Ry):
		temp = math.sqrt((Ry-By)**2 + (Rx-Bx)**2)
		LosX = (Rx-Bx)/(temp+0.000001)
		LosY = (Ry-By)/(temp+0.000001)
		BlueVecY = math.cos(heading*math.pi/180)
		BlueVecX = math.sin(heading*math.pi/180)

		vector_1 = [BlueVecX, BlueVecY]
		vector_2 = [LosX, LosY]

		unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
		unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
		dot_product = np.dot(unit_vector_1, unit_vector_2)
		angle = np.arccos(dot_product)

		#print("angle",angle*180/math.pi)
		return_val = angle*180/np.pi
		if(math.isnan(return_val)):
			return_val = 0
		return return_val

X = np.zeros((10**5,8)) #state space
Xl = np.zeros((10**5,8)) #state space when go to left 
Xr = np.zeros((10**5,8)) #state space when go to right 
Xf = np.zeros((10**5,8)) #state space when go to forward
J_vals = np.zeros((10**5))
feature_space = np.zeros((10**5, globals.FEATURE_NUMBER))
feature_space_left = np.zeros((10**5, globals.FEATURE_NUMBER))
feature_space_right = np.zeros((10**5, globals.FEATURE_NUMBER))
feature_space_forward = np.zeros((10**5, globals.FEATURE_NUMBER))




prev = time.time()
for i in range (10**5):
	if(i%1000 == 0):
		print(i)
	bx, by = random.randint(-100, 100), random.randint(-100, 100)
	rx, ry = random.randint(-100, 100), random.randint(-100, 100)

	blue_uav = ReinforcementModule([bx, by, 100], "blue")
	red_uav = AI([rx, ry, 100], "red")

	

	b_yaw  = blue_uav.yaw
	b_roll = blue_uav.roll

	r_yaw  = red_uav.yaw
	r_roll = red_uav.roll

	X[i] = [bx, by, b_yaw, b_roll, rx, ry, r_yaw, r_roll]

	estimated_red_action = red_uav.takeAction(blue_uav, False)
	for _ in range(1):
		red_uav.myApplyAction(estimated_red_action)

	Xf[i][4], Xf[i][5], Xf[i][6], Xf[i][7] = red_uav.position[0], red_uav.position[1], red_uav.yaw, red_uav.roll
	Xr[i][4], Xr[i][5], Xr[i][6], Xr[i][7] = red_uav.position[0], red_uav.position[1], red_uav.yaw, red_uav.roll
	Xl[i][4], Xl[i][5], Xl[i][6], Xl[i][7] = red_uav.position[0], red_uav.position[1], red_uav.yaw, red_uav.roll

	
	for _ in range(1):
		blue_uav.myApplyAction("left")

	ATA = getAng(blue_uav.yaw, blue_uav.position[0], blue_uav.position[1], red_uav.position[0], red_uav.position[1])
	AA = 180-getAng(red_uav.yaw, red_uav.position[0], red_uav.position[1], blue_uav.position[0], blue_uav.position[1])
	R = ((blue_uav.position[0] - red_uav.position[0])**2 + (blue_uav.position[1]-red_uav.position[1])**2)**0.5
	S = calculateS(AA, ATA, R, Rd = 3)
	features = blue_uav.createFeatureSpace(ATA, AA, R, red_uav.yaw, blue_uav.yaw).copy()
	feature_space_left[i] = features
	#print("leftLen: ",len(feature_space_left))
	Xl[i][0], Xl[i][1], Xl[i][2], Xl[i][3] = blue_uav.position[0], blue_uav.position[1], blue_uav.yaw, blue_uav.roll
	blue_uav.position[0], blue_uav.position[1], blue_uav.yaw, blue_uav.roll = bx, by, b_yaw, b_roll

	for _ in range(1):
		blue_uav.myApplyAction("right")

	ATA = getAng(blue_uav.yaw, blue_uav.position[0], blue_uav.position[1], red_uav.position[0], red_uav.position[1])
	AA = 180-getAng(red_uav.yaw, red_uav.position[0], red_uav.position[1], blue_uav.position[0], blue_uav.position[1])
	R = ((blue_uav.position[0] - red_uav.position[0])**2 + (blue_uav.position[1]-red_uav.position[1])**2)**0.5
	S = calculateS(AA, ATA, R, Rd = 3)
	features = blue_uav.createFeatureSpace(ATA,AA,R, red_uav.yaw, blue_uav.yaw).copy()
	feature_space_right[i] = features
	#print("rightLen: ",len(feature_space_right))

	Xr[i][0], Xr[i][1], Xr[i][2], Xr[i][3] = blue_uav.position[0], blue_uav.position[1], blue_uav.yaw, blue_uav.roll
	blue_uav.position[0], blue_uav.position[1], blue_uav.yaw, blue_uav.roll = bx, by, b_yaw, b_roll

	for _ in range(1):
		blue_uav.myApplyAction("forward")

	ATA = getAng(blue_uav.yaw, blue_uav.position[0], blue_uav.position[1], red_uav.position[0], red_uav.position[1])
	AA = 180-getAng(red_uav.yaw, red_uav.position[0], red_uav.position[1], blue_uav.position[0], blue_uav.position[1])
	R = ((blue_uav.position[0] - red_uav.position[0])**2 + (blue_uav.position[1]-red_uav.position[1])**2)**0.5
	S = calculateS(AA, ATA, R, Rd = 3)
	features = blue_uav.createFeatureSpace(ATA,AA,R, red_uav.yaw, blue_uav.yaw).copy()
	feature_space_forward[i] = features
	#print("forwardLen: ",len(feature_space_forward))

	Xf[i][0], Xf[i][1], Xf[i][2], Xf[i][3] = blue_uav.position[0], blue_uav.position[1], blue_uav.yaw, blue_uav.roll
	blue_uav.position[0], blue_uav.position[1], blue_uav.yaw, blue_uav.roll = bx, by, b_yaw, b_roll

	
	red_uav.position[0], red_uav.position[1], red_uav.yaw, red_uav.roll = rx, ry, r_yaw, r_roll


	ATA = getAng(b_yaw, bx, by, rx, ry)
	AA = 180-getAng(r_yaw, rx, ry, bx, by)
	R = ((bx-rx)**2 + (by-ry)**2)**0.5
	S = calculateS(AA, ATA, R, Rd = 3)
	features = blue_uav.createFeatureSpace(ATA,AA,R, r_yaw, b_yaw).copy()
	feature_space[i] = features
	#print("featureLen: ",len(feature_space))
	
	J_vals[i] = S
print(time.time()-prev)

print(feature_space)
Beta = (np.linalg.pinv(feature_space.T @ feature_space) @ feature_space.T) @ J_vals
#Beta = globals.Beta
print("Beta:",np.array2string(Beta, separator=', '))
print(time.time()-prev)


N = 250 #iteration number
actions = [Xl, Xr, Xf]
feature_list = [feature_space_left, feature_space_right, feature_space_forward]

for i in range(N):
	print("i:",i)
	prev = time.time()
	for j in range(10**5):
		bestJ = -100
		for k, action in enumerate(actions):
			
			bx, by, rx, ry = action[j][0], action[j][1], action[j][4], action[j][5]
			b_yaw, r_yaw = action[j][2], action[j][6]
			ATA = getAng(b_yaw, bx, by, rx, ry)
			AA = 180-getAng(r_yaw, rx, ry, bx, by)
			R = ((bx-rx)**2 + (by-ry)**2)**0.5
			g_val = GFunction(R, AA, ATA)
			############## feature space i 3 le bu sayede bir sonraki durumun feature spacesine bakabilmiş ol
			temp = 0.8*(feature_list[k][j,:] @ Beta) + g_val #0.8 discount factor
			##############
			if(temp > bestJ):
				bestJ = temp

		J_vals[j] = bestJ
	Beta = (np.linalg.pinv(feature_space.T @ feature_space) @ feature_space.T) @ J_vals
	print(np.array2string(Beta, separator=', '))

	"""if i%30==0:
		env = Environment(Beta=Beta)
		env.simulate_n(1)"""

print(np.array2string(Beta, separator=', '))

