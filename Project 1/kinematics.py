import numpy as np
import matplotlib.pyplot as plt
import math

## ------------------------------------ ##
## Data for Normal walking - kinematics ##
## ------------------------------------ ##

#data_normwalk = np.loadtxt("Project 1/walking.txt",skiprows=1)
data_normwalk = np.loadtxt("Project 1/walking.txt",skiprows=1)
# i had to change \ to / to run the code /Tor

## Right gait ##
## ---------- ##

tOnR = 218 # = frame 219 = line 219 - 1 because header
tOffR = 308 # = frame 308 = line 308 because header
tNormR = data_normwalk[tOnR:tOffR,1]

## Trunk angle : TRXO upper & TRXP lower ##

trxoRX = data_normwalk[tOnR:tOffR,33]
trxoRY = data_normwalk[tOnR:tOffR,34]

trxpRX = data_normwalk[tOnR:tOffR,36]
trxpRY = data_normwalk[tOnR:tOffR,37]

trunkangleR = np.zeros_like(tNormR)


for i in range(tOffR - tOnR):
    deltay = trxoRY[i] - trxpRY[i]
    deltax = trxoRX[i] - trxpRX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    trunkangleR[i] = 90 - angle_degrees

## Pelvis angle : PELP = upper & PELO = lower ##

peloRX = data_normwalk[tOnR:tOffR,3]
peloRY = data_normwalk[tOnR:tOffR,4]

pelpRX = data_normwalk[tOnR:tOffR,6]
pelpRY = data_normwalk[tOnR:tOffR,7]

pelvisangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay = pelpRY[i] - peloRY[i]
    deltax = pelpRX[i] - peloRX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    pelvisangleR[i] = 90 - angle_degrees

## Hip angle : between pelvis and thigh, thigh segment = hip to knee ##

hipRX = data_normwalk[tOnR:tOffR,18]
hipRY = data_normwalk[tOnR:tOffR,19]
kneeRX = data_normwalk[tOnR:tOffR,21]
kneeRY = data_normwalk[tOnR:tOffR,22]

hipangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay = hipRY[i] - kneeRY[i]
    deltax = hipRX[i] - kneeRX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    hipangleR[i] = angle_degrees - 90 + pelvisangleR[i]

## Knee angle : between shank and thigh, shank segment = knee to ankle ##

ankleRX = data_normwalk[tOnR:tOffR,24]
ankleRY = data_normwalk[tOnR:tOffR,25]

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

footRX = data_normwalk[tOnR:tOffR,30]
footRY = data_normwalk[tOnR:tOffR,31]

ankleangleR = np.zeros_like(tNormR)

for i in range(tOffR - tOnR):
    deltay_shank = kneeRY[i] - ankleRY[i]
    deltax_shank = kneeRX[i] - ankleRX[i]
    angle_degrees_shank = 90 - math.degrees(math.atan2(deltay_shank,deltax_shank)) # blue angle drawing with square
    print(angle_degrees_shank)
    deltay = footRY[i] - ankleRY[i]
    deltax = footRX[i] - ankleRX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    ankleangleR[i] = angle_degrees + angle_degrees_shank + 5

## Left gait ##
## --------- ##

tOnL = 261 # = frame 262 = line 262 - 1 because header
tOffL = 351 # = frame 351 = line 351 because header
tNormL = data_normwalk[tOnL:tOffL,1]

## Trunk angle : TRXO upper & TRXP lower ##

trxoLX = data_normwalk[tOnL:tOffL,33]
trxoLY = data_normwalk[tOnL:tOffL,34]

trxpLX = data_normwalk[tOnL:tOffL,36]
trxpLY = data_normwalk[tOnL:tOffL,37]

trunkangleL = np.zeros_like(tNormL)


for i in range(tOffL - tOnL):
    deltay = trxoLY[i] - trxpLY[i]
    deltax = trxoLX[i] - trxpLX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    trunkangleL[i] = 90 - angle_degrees

## Pelvis angle : PELP = upper & PELO = lower ##

peloLX = data_normwalk[tOnL:tOffL,3]
peloLY = data_normwalk[tOnL:tOffL,4]

pelpLX = data_normwalk[tOnL:tOffL,6]
pelpLY = data_normwalk[tOnL:tOffL,7]

pelvisangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay = pelpLY[i] - peloLY[i]
    deltax = pelpLX[i] - peloLX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    pelvisangleL[i] = 90 - angle_degrees

## Hip angle : between pelvis and thigh, thigh segment = hip to knee ##

hipLX = data_normwalk[tOnL:tOffL,12]
hipLY = data_normwalk[tOnL:tOffL,13]
kneeLX = data_normwalk[tOnL:tOffL,9]
kneeLY = data_normwalk[tOnL:tOffL,10]

hipangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay = hipLY[i] - kneeLY[i]
    deltax = hipLX[i] - kneeLX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    hipangleL[i] = angle_degrees - 90 + pelvisangleL[i]

## Knee angle : between shank and thigh, shank segment = knee to ankle ##

ankleLX = data_normwalk[tOnL:tOffL,15]
ankleLY = data_normwalk[tOnL:tOffL,16]

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

footLX = data_normwalk[tOnL:tOffL,27]
footLY = data_normwalk[tOnL:tOffL,28]

ankleangleL = np.zeros_like(tNormL)

for i in range(tOffL - tOnL):
    deltay_shank = kneeLY[i] - ankleLY[i]
    deltax_shank = kneeLX[i] - ankleLX[i]
    angle_degrees_shank = 90 - math.degrees(math.atan2(deltay_shank,deltax_shank)) # blue angle drawing with square
    deltay = footLY[i] - ankleLY[i]
    deltax = footLX[i] - ankleLX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    ankleangleL[i] = angle_degrees + angle_degrees_shank + 5

## Graph for comparison of Left and Right Gait Normal walking ##
## ---------------------------------------------------------- ##

# # Trunk angle comparison #

# fig, ax = plt.subplots()
# fig.set_size_inches(15, 8)
 
# ax.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,trunkangleR,c='mediumblue',label='Right Gait')
# ax.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
# ax.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,trunkangleL, c='darkorange',label = 'Left Gait')
# ax.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Trunk angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -4, 'Posterior tilt', color='red',fontsize=18, rotation=90)
# plt.text(-11, -0.3, 'Anterior tilt', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax.vlines(x=0.2, ymin=0, ymax=1, color='green',linewidth=12)
# ax.vlines(x=0.2, ymin=-5, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-4,0.5])
# plt.legend(fontsize= 20)

# # Pelvis angle comparison #

# fig1, ax1 = plt.subplots()
# fig1.set_size_inches(15, 8)
 
# ax1.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,pelvisangleR,c='mediumblue',label='Right Gait')
# ax1.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
# ax1.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,pelvisangleL, c='darkorange',label = 'Left Gait')
# ax1.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Pelvis angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -1, 'Posterior tilt', color='red',fontsize=18, rotation=90)
# plt.text(-11, 11, 'Anterior tilt', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax1.vlines(x=0.2, ymin=0, ymax=15, color='green',linewidth=12)
# ax1.vlines(x=0.2, ymin=-0.5, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-0.5,15])
# plt.legend(fontsize= 20)

# # Hip angle comparison #

# fig2, ax2 = plt.subplots()
# fig2.set_size_inches(15, 8)
 
# ax2.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,hipangleR,c='mediumblue',label='Right Gait')
# ax2.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
# ax2.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,hipangleL, c='darkorange',label = 'Left Gait')
# ax2.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Hip angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -12, 'Extension', color='red',fontsize=18, rotation=90)
# plt.text(-11, 25, 'Flexion', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax2.vlines(x=0.2, ymin=0, ymax=40, color='green',linewidth=12)
# ax2.vlines(x=0.2, ymin=-15, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-15,40])
# plt.legend(fontsize= 20)

# # Knee angle comparison #

# fig3, ax3 = plt.subplots()
# fig3.set_size_inches(15, 8)
 
# ax3.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,kneeangleR,c='mediumblue',label='Right Gait')
# ax3.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
# ax3.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,kneeangleL, c='darkorange',label = 'Left Gait')
# ax3.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Knee angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -7, 'Hyperextension', color='red',fontsize=18, rotation=90)
# plt.text(-11, 40, 'Flexion', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax3.vlines(x=0.2, ymin=0, ymax=55, color='green',linewidth=12)
# ax3.vlines(x=0.2, ymin=-5, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-5,55])
# plt.legend(fontsize= 20)

# Ankle angle comparison #

fig4, ax4 = plt.subplots()
fig4.set_size_inches(15, 8)
 
ax4.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,ankleangleR,c='mediumblue',label='Right Gait')
ax4.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
ax4.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,ankleangleL, c='darkorange',label = 'Left Gait')
ax4.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')

plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
plt.ylabel("Ankle angle [°]",fontsize=20)
plt.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

plt.text(-11, -20, 'Plantarflexion', color='red',fontsize=18, rotation=90)
plt.text(-11, 5, 'Dorsiflexion', color='green',fontsize=18, rotation=90)
plt.axhline(y=0, color='k')

ax4.vlines(x=0.2, ymin=0, ymax=15, color='green',linewidth=12)
ax4.vlines(x=0.2, ymin=-20, ymax=0, color='red',linewidth=12)

plt.axis([0,100,-20,15])
plt.legend(fontsize= 20)


## Writing in files ##

with open ('Project 1/kinematics_norm_R.txt', 'w') as file:  
    for i in range(len(pelvisangleR)):
       file.write(f"{trunkangleR[i]}\t{pelvisangleR[i]}\t{hipangleR[i]}\t{kneeangleR[i]}\t{ankleangleR[i]}\n") 

with open ('Project 1/kinematics_norm_L.txt', 'w') as file:  
    for i in range(len(pelvisangleL)):
       file.write(f"{trunkangleL[i]}\t{pelvisangleL[i]}\t{hipangleL[i]}\t{kneeangleL[i]}\t{ankleangleL[i]}\n") 

## ------------------------------------ ##
## Data for Crouch walking - kinematics ##
## ------------------------------------ ##

data_crouchwalk = np.loadtxt("Project 1/crouch.txt",skiprows=1)

## Right gait ##
## ---------- ##

tOn = 421 # = frame 422 = line 422 - 1 because header
tOff = 526 # = frame 526 = line 526 because header
tCrouchR = data_crouchwalk[tOn:tOff,1]

## Trunk angle : TRXO upper & TRXP lower ##

trxoCX = data_crouchwalk[tOn:tOff,33]
trxoCY = data_crouchwalk[tOn:tOff,34]

trxpCX = data_crouchwalk[tOn:tOff,36]
trxpCY = data_crouchwalk[tOn:tOff,37]

trunkangleC = np.zeros_like(tCrouchR)


for i in range(tOff - tOn):
    deltay = trxoCY[i] - trxpCY[i]
    deltax = trxoCX[i] - trxpCX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    trunkangleC[i] = 90 - angle_degrees

## Pelvis angle : PELP = upper & PELO = lower ##

peloCX = data_crouchwalk[tOn:tOff,3]
peloCY = data_crouchwalk[tOn:tOff,4]

pelpCX = data_crouchwalk[tOn:tOff,6]
pelpCY = data_crouchwalk[tOn:tOff,7]

pelvisangleC = np.zeros_like(tCrouchR)

for i in range(tOff - tOn):
    deltay = pelpCY[i] - peloCY[i]
    deltax = pelpCX[i] - peloCX[i]
    angle_radians = math.atan2(deltay, deltax)
    angle_degrees = math.degrees(angle_radians)
    # Convert the angle to be with respect to the vertical line
    pelvisangleC[i] = 90 - angle_degrees

## Hip angle : between pelvis and thigh, thigh segment = hip to knee ##

hipCX = data_crouchwalk[tOn:tOff,18]
hipCY = data_crouchwalk[tOn:tOff,19]
kneeCX = data_crouchwalk[tOn:tOff,21]
kneeCY = data_crouchwalk[tOn:tOff,22]

hipangleC = np.zeros_like(tCrouchR)

for i in range(tOff - tOn):
    deltay = hipCY[i] - kneeCY[i]
    deltax = hipCX[i] - kneeCX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    hipangleC[i] = angle_degrees - 90 + pelvisangleC[i]

## Knee angle : between shank and thigh, shank segment = knee to ankle ##

ankleCX = data_crouchwalk[tOn:tOff,24]
ankleCY = data_crouchwalk[tOn:tOff,25]

kneeangleC = np.zeros_like(tCrouchR)

for i in range(tOff - tOn):
    deltay_hip = hipCY[i] - kneeCY[i]
    deltax_hip = hipCX[i] - kneeCX[i]
    angle_degrees_hip = math.degrees(math.atan2(deltay_hip,deltax_hip)) - 90 # blue angle drawing *
    deltay = kneeCY[i] - ankleCY[i]
    deltax = kneeCX[i] - ankleCX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    kneeangleC[i] = 90 - angle_degrees + angle_degrees_hip

## Ankle angle : between ankle and shank, ankle segment = ankle to toe

footCX = data_crouchwalk[tOn:tOff,30]
footCY = data_crouchwalk[tOn:tOff,31]

ankleangleC = np.zeros_like(tCrouchR)

for i in range(tOff - tOn):
    deltay_shank = kneeCY[i] - ankleCY[i]
    deltax_shank = kneeCX[i] - ankleCX[i]
    angle_degrees_shank = 90 - math.degrees(math.atan2(deltay_shank,deltax_shank)) # blue angle drawing with square
    deltay = footCY[i] - ankleCY[i]
    deltax = footCX[i] - ankleCX[i]
    angle_radians = math.atan2(deltay,deltax)
    angle_degrees = math.degrees(angle_radians)
    ankleangleC[i] = angle_degrees + angle_degrees_shank + 5

## Graph for comparison of Right Gait Normal and Crouch walking ##
## ---------------------------------------------------------- ##

# # Trunk angle comparison #

# fig0, ax0 = plt.subplots()
# fig0.set_size_inches(15, 8)
 
# ax0.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,trunkangleR,c='mediumblue',label='Normal Gait')
# ax0.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off normal gait')
# ax0.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,trunkangleC, c='darkorange',label = 'Crouch Gait')
# ax0.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkorange',linestyle='dotted',label='toe-off crouch gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Trunk angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -5, 'Posterior tilt', color='red',fontsize=18, rotation=90)
# plt.text(-11, 14, 'Anterior tilt', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax0.vlines(x=0.2, ymin=0, ymax=20, color='green',linewidth=12)
# ax0.vlines(x=0.2, ymin=-5, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-5,20])
# plt.legend(fontsize= 20)

# # Pelvis angle comparison #

# fig10, ax10 = plt.subplots()
# fig10.set_size_inches(15, 8)
 
# ax10.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,pelvisangleR,c='mediumblue',label='Normal Gait')
# ax10.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off normal gait')
# ax10.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,pelvisangleC, c='darkorange',label = 'Crouch Gait')
# ax10.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkorange',linestyle='dotted',label='toe-off crouch gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Pelvis angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -1, 'Posterior tilt', color='red',fontsize=18, rotation=90)
# plt.text(-11, 20, 'Anterior tilt', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax10.vlines(x=0.2, ymin=0, ymax=30, color='green',linewidth=12)
# ax10.vlines(x=0.2, ymin=-1, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-1,30])
# plt.legend(fontsize= 20)

# # Hip angle comparison #

# fig20, ax20 = plt.subplots()
# fig20.set_size_inches(15, 8)
 
# ax20.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,hipangleR,c='mediumblue',label='Normal Gait')
# ax20.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off normal gait')
# ax20.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,hipangleC, c='darkorange',label = 'Crouch Gait')
# ax20.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkorange',linestyle='dotted',label='toe-off crouch gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Hip angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -15, 'Extension', color='red',fontsize=18, rotation=90)
# plt.text(-11, 50, 'Flexion', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax20.vlines(x=0.2, ymin=0, ymax=70, color='green',linewidth=12)
# ax20.vlines(x=0.2, ymin=-15, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-15,70])
# plt.legend(fontsize= 20)

# # Knee angle comparison #

# fig30, ax30 = plt.subplots()
# fig30.set_size_inches(15, 8)
 
# ax30.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,kneeangleR,c='mediumblue',label='Normal Gait')
# ax30.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off normal gait')
# ax30.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,kneeangleC, c='darkorange',label = 'Crouch Gait')
# ax30.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkorange',linestyle='dotted',label='toe-off crouch gait')

# plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
# plt.ylabel("Knee angle [°]",fontsize=20)
# plt.grid('True')

# for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#   tickLabel.set_fontsize(16)

# plt.text(-11, -7, 'Hyperextension', color='red',fontsize=18, rotation=90)
# plt.text(-11, 50, 'Flexion', color='green',fontsize=18, rotation=90)
# plt.axhline(y=0, color='k')

# ax30.vlines(x=0.2, ymin=0, ymax=70, color='green',linewidth=12)
# ax30.vlines(x=0.2, ymin=-5, ymax=0, color='red',linewidth=12)

# plt.axis([0,100,-5,70])
# plt.legend(fontsize= 20)

# Ankle angle comparison #

fig40, ax40 = plt.subplots()
fig40.set_size_inches(15, 8)
 
ax40.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,ankleangleR,c='mediumblue',label='Normal Gait')
ax40.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off normal gait')
ax40.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,ankleangleC, c='darkorange',label = 'Crouch Gait')
ax40.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkorange',linestyle='dotted',label='toe-off crouch gait')

plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
plt.ylabel("Ankle angle [°]",fontsize=20)
plt.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

plt.text(-11, -20, 'Plantarflexion', color='red',fontsize=18, rotation=90)
plt.text(-11, 15, 'Dorsiflexion', color='green',fontsize=18, rotation=90)
plt.axhline(y=0, color='k')

ax40.vlines(x=0.2, ymin=0, ymax=30, color='green',linewidth=12)
ax40.vlines(x=0.2, ymin=-20, ymax=0, color='red',linewidth=12)

plt.axis([0,100,-20,30])
plt.legend(fontsize= 20)

plt.show()

# Writing in file 

with open ('Project 1/kinematics_crouch_R.txt', 'w') as file:  
    for i in range(len(pelvisangleC)):
       file.write(f"{trunkangleC[i]}\t{pelvisangleC[i]}\t{hipangleC[i]}\t{kneeangleC[i]}\t{ankleangleC[i]}\n") 