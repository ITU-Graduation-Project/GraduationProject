
def GFunction(R, AA, ATA):
	"""
		R: Euclidean distance between aircraft
		AA: Aspact Angle 
		ATA: Antenna Train Angle
	"""

	if(0.1 < R < 3.0 and abs(AA) < 60 and abs(ATA) < 30):
		g = 1.0
	else:
		g = 0.0

	return g
