import numpy as np
import matplotlib.pyplot as plt
import math

## ------------------------------------ ##
## Data for Normal walking - kinematics ##
## ------------------------------------ ##

data_normwak = np.loadtxt("walking.txt",skiprows=1)

## Right gait ##
## ---------- ##

tOnR = 218 # = frame 219 = line 219 - 1 because header
tOffR = 308 # = frame 308 = line 308 because header
tNormR = data_normwak[tOnR:tOffR,1]

## Trunk angle : TRXO upper & TRXP lower ##

trxoRX = data_normwak[tOnR:tOffR,33]
trxoRY = data_normwak[tOnR:tOffR,34]

trxpRX = data_normwak[tOnR:tOffR,36]
trxpRY = data_normwak[tOnR:tOffR,37]

trunkangleR = np.zeros_like(tNormR)


for i in range(tOffR - tOnR):
    deltay = trxoRY[i] - trxpRY[i]
    deltax = trxoRX[i] - trxpRX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    trunkangleR[i] = 90 - angle_degrees

## Pelvis angle : PELP = upper & PELO = lower ##

peloRX = data_normwak[tOnR:tOffR,3]
peloRY = data_normwak[tOnR:tOffR,4]

pelpRX = data_normwak[tOnR:tOffR,6]
pelpRY = data_normwak[tOnR:tOffR,7]

pelvisangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay = pelpRY[i] - peloRY[i]
    deltax = pelpRX[i] - peloRX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    pelvisangleR[i] = 90 - angle_degrees

## Hip angle : between pelvis and thigh, thigh segment = hip to knee ##

hipRX = data_normwak[tOnR:tOffR,18]
hipRY = data_normwak[tOnR:tOffR,19]
kneeRX = data_normwak[tOnR:tOffR,21]
kneeRY = data_normwak[tOnR:tOffR,22]

hipangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay = hipRY[i] - kneeRY[i]
    deltax = hipRX[i] - kneeRX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    hipangleR[i] = angle_degrees - 90 + pelvisangleR[i]

## Knee angle : between shank and thigh, shank segment = knee to ankle ##

ankleRX = data_normwak[tOnR:tOffR,24]
ankleRY = data_normwak[tOnR:tOffR,25]

kneeangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay_hip = hipRY[i] - kneeRY[i]
    deltax_hip = hipRX[i] - kneeRX[i]
    angle_degrees_hip = math.degrees(math.atan2(deltay_hip,deltax_hip)) - 90 # blue angle drawing *
    deltay = kneeRY[i] - ankleRY[i]
    deltax = kneeRX[i] - ankleRX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    kneeangleR[i] = 90 - angle_degrees + angle_degrees_hip

## Ankle angle : between ankle and shank, ankle segment = ankle to toe

footRX = data_normwak[tOnR:tOffR,30]
footRY = data_normwak[tOnR:tOffR,31]

ankleangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay_shank = kneeRY[i] - ankleRY[i]
    deltax_shank = kneeRX[i] - ankleRX[i]
    angle_degrees_shank = 90 - math.degrees(math.atan2(deltay_shank,deltax_shank)) # blue angle drawing with square
    deltay = footRY[i] - ankleRY[i]
    deltax = footRX[i] - ankleRX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    ankleangleR[i] = angle_degrees - angle_degrees_shank

## Left gait ##
## --------- ##

tOnL = 261 # = frame 262 = line 262 - 1 because header
tOffL = 351 # = frame 351 = line 351 because header
tNormL = data_normwak[tOnL:tOffL,1]

## Trunk angle : TRXO upper & TRXP lower ##

trxoLX = data_normwak[tOnL:tOffL,33]
trxoLY = data_normwak[tOnL:tOffL,34]

trxpLX = data_normwak[tOnL:tOffL,36]
trxpLY = data_normwak[tOnL:tOffL,37]

trunkangleL = np.zeros_like(tNormL)


for i in range(tOffL - tOnL):
    deltay = trxoLY[i] - trxpLY[i]
    deltax = trxoLX[i] - trxpLX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    trunkangleL[i] = 90 - angle_degrees

## Pelvis angle : PELP = upper & PELO = lower ##

peloLX = data_normwak[tOnL:tOffL,3]
peloLY = data_normwak[tOnL:tOffL,4]

pelpLX = data_normwak[tOnL:tOffL,6]
pelpLY = data_normwak[tOnL:tOffL,7]

pelvisangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay = pelpLY[i] - peloLY[i]
    deltax = pelpLX[i] - peloLX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    pelvisangleL[i] = 90 - angle_degrees

## Hip angle : between pelvis and thigh, thigh segment = hip to knee ##

hipLX = data_normwak[tOnL:tOffL,12]
hipLY = data_normwak[tOnL:tOffL,13]
kneeLX = data_normwak[tOnL:tOffL,9]
kneeLY = data_normwak[tOnL:tOffL,10]

hipangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay = hipLY[i] - kneeLY[i]
    deltax = hipLX[i] - kneeLX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    hipangleL[i] = angle_degrees - 90 + pelvisangleL[i]

## Knee angle : between shank and thigh, shank segment = knee to ankle ##

ankleLX = data_normwak[tOnL:tOffL,15]
ankleLY = data_normwak[tOnL:tOffL,16]

kneeangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay_hip = hipLY[i] - kneeLY[i]
    deltax_hip = hipLX[i] - kneeLX[i]
    angle_degrees_hip = math.degrees(math.atan2(deltay_hip,deltax_hip)) - 90 # blue angle drawing *
    deltay = kneeLY[i] - ankleLY[i]
    deltax = kneeLX[i] - ankleLX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    kneeangleL[i] = 90 - angle_degrees + angle_degrees_hip

## Ankle angle : between ankle and shank, ankle segment = ankle to toe

footLX = data_normwak[tOnL:tOffL,27]
footLY = data_normwak[tOnL:tOffL,28]

ankleangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay_shank = kneeLY[i] - ankleLY[i]
    deltax_shank = kneeLX[i] - ankleLX[i]
    angle_degrees_shank = 90 - math.degrees(math.atan2(deltay_shank,deltax_shank)) # blue angle drawing with square
    deltay = footLY[i] - ankleLY[i]
    deltax = footLX[i] - ankleLX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    ankleangleL[i] = angle_degrees - angle_degrees_shank

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
 
ax.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0]),trunkangleR,c='mediumblue',label='Right Gait')
ax.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0]),trunkangleL, c='darkorange',label = 'Left Gait')

plt.legend()
plt.xlabel("Time [s]",fontsize=20)
plt.ylabel("Trunk angle [°]",fontsize=20)
ax.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

#plt.axis([0.0,1.0,0.1,18])
plt.legend(fontsize= 20)
plt.show()
