import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

##Computing lengths

def analyze_joint_lengths(file_paths):
    # Lists to store lengths for both sides
    foot_lengths, shank_lengths, thigh_lengths = [], [], []

    for file_path in file_paths:
        # Load marker trajectory data
        data_trc = pd.read_csv(file_path)

        # Conversion factor
        to_meters = 1 / 1000

        # Extract relevant columns for both left and right sides
        joint_columns = ['RTOO_Y', 'RTOO_Z', 'RAJC_Y', 'RAJC_Z',
                         'RKJC_Y', 'RKJC_Z', 'RHJC_Y', 'RHJC_Z',
                         'LTOO_Y', 'LTOO_Z', 'LAJC_Y', 'LAJC_Z',
                         'LKJC_Y', 'LKJC_Z', 'LHJC_Y', 'LHJC_Z']

        # Extract joint coordinates for both left and right sides
        joints = data_trc[joint_columns] * to_meters

        # Filter out rows where any of the coordinates is zero for both sides
        non_zero_rows = joints[~(joints == 0).any(axis=1)]

        # Function to calculate distance between two points in 3D space
        def calculate_distance(point1, point2):
            return np.linalg.norm(point1 - point2, axis=1)

        # Calculate lengths for both left and right sides
        foot_length_right = calculate_distance(non_zero_rows[['RTOO_Y', 'RTOO_Z']].values,
                                         non_zero_rows[['RAJC_Y', 'RAJC_Z']].values)
        shank_length_right = calculate_distance(non_zero_rows[['RAJC_Y', 'RAJC_Z']].values,
                                          non_zero_rows[['RKJC_Y', 'RKJC_Z']].values)
        thigh_length_right = calculate_distance(non_zero_rows[['RKJC_Y', 'RKJC_Z']].values,
                                          non_zero_rows[['RHJC_Y', 'RHJC_Z']].values)

        # Include left side calculations
        foot_length_left = calculate_distance(non_zero_rows[['LTOO_Y', 'LTOO_Z']].values,
                                              non_zero_rows[['LAJC_Y', 'LAJC_Z']].values)
        shank_length_left = calculate_distance(non_zero_rows[['LAJC_Y', 'LAJC_Z']].values,
                                               non_zero_rows[['LKJC_Y', 'LKJC_Z']].values)
        thigh_length_left = calculate_distance(non_zero_rows[['LKJC_Y', 'LKJC_Z']].values,
                                               non_zero_rows[['LHJC_Y', 'LHJC_Z']].values)

        # Append lengths to lists
        foot_lengths.extend(foot_length_right)
        shank_lengths.extend(shank_length_right)
        thigh_lengths.extend(thigh_length_right)

        foot_lengths.extend(foot_length_left)
        shank_lengths.extend(shank_length_left)
        thigh_lengths.extend(thigh_length_left)

    # Display the overall averages
    # print("\nOverall Averages:")
    # print(f"Average Foot Length: {np.mean(foot_lengths)} meters")
    # print(f"Average Shank Length: {np.mean(shank_lengths)} meters")
    # print(f"Average Thigh Length: {np.mean(thigh_lengths)} meters")
    return np.mean(foot_lengths),np.mean(shank_lengths),np.mean(thigh_lengths)

# Replace these with the actual paths to your files
file_paths = ['Project 1/crouch.csv', 'Project 1/walking.csv']
foot_length,shank_length,thigh_length=analyze_joint_lengths(file_paths)


# Constants and anthropometric data
height = 1.680  # in m
weight = 71.5  # in kg
g = 9.81 # m/s²

# Anthropometric data for thigh
lt = thigh_length  # in m
thigh_mass_percentage = 0.1  # Replace with actual percentage
thigh_com_percentage = 0.433  # Replace with actual percentage

# Anthropometric data for shank
ls = shank_length  # in m
shank_mass_percentage = 0.0465  # Replace with actual percentage
shank_com_percentage = 0.433  # Replace with actual percentage

# Anthropometric data for foot
lf = foot_length  # in m
foot_mass_percentage = 0.0145  # Replace with actual percentage
foot_com_percentage = 0.5  # Replace with actual percentage

pelvis_mass_ratio = 0.142
pelvis_COM_ratio = 0.25

mt = weight * thigh_mass_percentage 
ms = weight * shank_mass_percentage 
mf = weight * foot_mass_percentage 
mp = weight * pelvis_mass_ratio

dH = lt * thigh_com_percentage
dK = ls * shank_com_percentage
dA = lf * foot_com_percentage
pelvis_COM = height * pelvis_COM_ratio

If = 0.475 * lf
Is = 0.302 * ls
It = 0.323 * lt

# Function to calculate position of center of mass

def COM(ax, ay, bx, by, da):
    angle_AB = math.atan2(by - ay, bx - ax)
    Cx = ax + da * math.cos(angle_AB)  # Calculate x-coordinate at distance da from A
    Cy = ay + da * math.sin(angle_AB)  # Calculate y-coordinate at distance da from A
    return Cx, Cy

# Function to calculate distance vector 

def calculate_distance_vector(ax, ay, bx, by):
    distance_vector_x = bx - ax
    distance_vector_y = by - ay
    return distance_vector_x, distance_vector_y


## ---------------------------------- ##
## Data for Normal walking - kinetics ##
## ---------------------------------- ##

data_normwalk = np.loadtxt("Project 1/walking.txt",skiprows=1)
angles_normR = np.loadtxt("Project 1/kinematics_norm_R.txt")
angles_normL = np.loadtxt("Project 1/kinematics_norm_L.txt")
forces_norm = np.loadtxt("Project 1/walking_FP.txt",skiprows=1)

## Right gait ##

tOnR = 218 # = frame 219 = line 219 - 1 because header
tOffR = 308 # = frame 308 = line 308 because header
tNormR = data_normwalk[tOnR:tOffR,1]

F_gR_X_tot = forces_norm[:,11]
F_gR_Y_tot = forces_norm[:,12]
F_x_Rtot = forces_norm[:,8]/1000
F_y_Rtot = forces_norm[:,9]/1000

F_gR_X_s = F_gR_X_tot[::10]
F_gR_Y_s = F_gR_Y_tot[::10]
F_x_Rs = F_x_Rtot[::10]
F_y_Rs = F_y_Rtot[::10]

F_gR_X = F_gR_X_s[tOnR:tOffR]
F_gR_Y = F_gR_Y_s[tOnR:tOffR]
F_x_R = F_x_Rs[tOnR:tOffR]
F_y_R = F_y_Rs[tOnR:tOffR]


ankleRX = data_normwalk[tOnR:tOffR,24]/1000
ankleRY = data_normwalk[tOnR:tOffR,25]/1000
footRX = data_normwalk[tOnR:tOffR,30]/1000
footRY = data_normwalk[tOnR:tOffR,31]/1000
hipRX = data_normwalk[tOnR:tOffR,18]/1000
hipRY = data_normwalk[tOnR:tOffR,19]/1000
kneeRX = data_normwalk[tOnR:tOffR,21]/1000
kneeRY = data_normwalk[tOnR:tOffR,22]/1000


# Foot equations

footangleR = np.radians(angles_normR[:,4])

CfxR = np.zeros_like(ankleRX)
CfyR = np.zeros_like(ankleRY)
for i in range(len(ankleRX)):
    CfxR[i], CfyR[i] = COM(ankleRX[i],ankleRY[i],footRX[i],footRY[i],dA)
vfRx = np.gradient(CfxR,tNormR)
vfRy = np.gradient(CfyR,tNormR)
afRx = np.gradient(vfRx,tNormR)
afRy = np.gradient(vfRy,tNormR)
omegafR = np.gradient(footangleR,tNormR)
alphafR = np.gradient(omegafR,tNormR)
dA_xR, dA_yR = calculate_distance_vector(CfxR,CfyR,ankleRX,ankleRY)
dG_xR,dG_yR = calculate_distance_vector(CfxR,CfyR,F_x_R,F_y_R)

F_Ax_R = mf*afRx - F_gR_X
F_Ay_R = mf*afRy + mf*g - F_gR_Y
M_A_r = If*alphafR - dA_xR*F_Ay_R + dA_yR*F_Ax_R - dG_xR*F_gR_Y + dG_yR*F_gR_X

# Shank equations

shankangleR = np.radians(angles_normR[:,3])
CsxR = np.zeros_like(kneeRX)
CsyR = np.zeros_like(kneeRY)
for i in range(len(kneeRX)):
    CsxR[i], CsyR[i] = COM(kneeRX[i],kneeRY[i],ankleRX[i],ankleRY[i],dK)
vsRx = np.gradient(CsxR,tNormR)
vsRy = np.gradient(CsyR,tNormR)
asRx = np.gradient(vsRx,tNormR)
asRy = np.gradient(vsRy,tNormR)
omegasR = np.gradient(shankangleR,tNormR)
alphasR = np.gradient(omegasR,tNormR)
dK_xR, dK_yR = calculate_distance_vector(CsxR,CsyR,kneeRX,kneeRY)
lsdK_xR,lsdK_yR = calculate_distance_vector(CsxR,CsyR,ankleRX,ankleRY)

F_Kx_R = ms*asRx - F_Ax_R
F_Ky_R = ms*asRy + ms*g + F_Ay_R
M_K_r = Is*alphasR - dK_xR*F_Ky_R + dK_yR*F_Kx_R + M_A_r - lsdK_yR*F_Ax_R + lsdK_xR*F_Ay_R

# Thigh equations

thighangleR = np.radians(angles_normR[:,2])
CtxR = np.zeros_like(kneeRX)
CtyR = np.zeros_like(kneeRY)
for i in range(len(kneeRX)):
    CtxR[i], CtyR[i] = COM(hipRX[i],hipRY[i],kneeRX[i],kneeRY[i],dH)
vtRx = np.gradient(CtxR,tNormR)
vtRy = np.gradient(CtyR,tNormR)
atRx = np.gradient(vtRx,tNormR)
atRy = np.gradient(vtRy,tNormR)
omegatR = np.gradient(thighangleR,tNormR)
alphatR = np.gradient(omegatR,tNormR)
dH_xR, dH_yR = calculate_distance_vector(CtxR,CtyR,hipRX,hipRY)
ltdH_xR,ltdH_yR = calculate_distance_vector(CtxR,CtyR,kneeRX,kneeRY)

F_Hx_R = mt*atRx - F_Kx_R
F_Hy_R = mt*atRy + mt*g + F_Ky_R
M_H_r = It*alphatR - dH_xR*F_Hy_R + dH_yR*F_Hx_R + M_K_r - ltdH_yR*F_Kx_R + ltdH_xR*F_Ky_R

# Power

## Left Gait ##

tOnL = 261 # = frame 262 = line 262 - 1 because header
tOffL = 351 # = frame 351 = line 351 because header
tNormL = data_normwalk[tOnL:tOffL,1]

F_gL_X_tot = forces_norm[:,5]
F_gL_Y_tot = forces_norm[:,6]
F_x_Ltot = forces_norm[:,2]/1000
F_y_Ltot = forces_norm[:,3]/1000

F_gL_X_s = F_gL_X_tot[::10]
F_gL_Y_s = F_gL_Y_tot[::10]
F_x_Ls = F_x_Ltot[::10]
F_y_Ls = F_y_Ltot[::10]

F_gL_X = F_gL_X_s[tOnL:tOffL]
F_gL_Y = F_gL_Y_s[tOnL:tOffL]
F_x_L = F_x_Ls[tOnL:tOffL]
F_y_L = F_y_Ls[tOnL:tOffL]


ankleLX = data_normwalk[tOnL:tOffL,15]/1000
ankleLY = data_normwalk[tOnL:tOffL,16]/1000
footLX = data_normwalk[tOnL:tOffL,27]/1000
footLY = data_normwalk[tOnL:tOffL,28]/1000
hipLX = data_normwalk[tOnL:tOffL,12]/1000
hipLY = data_normwalk[tOnL:tOffL,13]/1000
kneeLX = data_normwalk[tOnL:tOffL,9]/1000
kneeLY = data_normwalk[tOnL:tOffL,10]/1000

# Foot equations

footangleL = np.radians(angles_normL[:,4])

CfxL = np.zeros_like(ankleLX)
CfyL = np.zeros_like(ankleLY)
for i in range(len(ankleLX)):
    CfxL[i], CfyL[i] = COM(ankleLX[i],ankleLY[i],footLX[i],footLY[i],dA)
vfLx = np.gradient(CfxL,tNormL)
vfLy = np.gradient(CfyL,tNormL)
afLx = np.gradient(vfLx,tNormL)
afLy = np.gradient(vfLy,tNormL)
omegafL = np.gradient(footangleL,tNormL)
alphafL = np.gradient(omegafL,tNormL)
dA_xL, dA_yL = calculate_distance_vector(CfxL,CfyL,ankleLX,ankleLY)
dG_xL,dG_yL = calculate_distance_vector(CfxL,CfyL,F_x_L,F_y_L)

F_Ax_L = mf*afLx - F_gL_X
F_Ay_L = mf*afLy + mf*g - F_gL_Y
M_A_l = If*alphafL - dA_xL*F_Ay_L + dA_yL*F_Ax_L - dG_xL*F_gL_Y + dG_yL*F_gL_X

# Shank equations

shankangleL = np.radians(angles_normL[:,3])
CsxL = np.zeros_like(kneeLX)
CsyL = np.zeros_like(kneeLY)
for i in range(len(kneeLX)):
    CsxL[i], CsyL[i] = COM(kneeLX[i],kneeLY[i],ankleLX[i],ankleLY[i],dK)
vsLx = np.gradient(CsxL,tNormL)
vsLy = np.gradient(CsyL,tNormL)
asLx = np.gradient(vsLx,tNormR)
asLy = np.gradient(vsLy,tNormL)
omegasL = np.gradient(shankangleL,tNormL)
alphasL = np.gradient(omegasL,tNormL)
dK_xL, dK_yL = calculate_distance_vector(CsxL,CsyL,kneeLX,kneeLY)
lsdK_xL,lsdK_yL = calculate_distance_vector(CsxL,CsyL,ankleLX,ankleLY)

F_Kx_L = ms*asLx - F_Ax_L
F_Ky_L = ms*asLy + ms*g + F_Ay_L
M_K_l = Is*alphasL - dK_xL*F_Ky_L + dK_yL*F_Kx_L + M_A_l - lsdK_yL*F_Ax_L + lsdK_xL*F_Ay_L

# Thigh equations

thighangleL = np.radians(angles_normL[:,2])
CtxL = np.zeros_like(kneeLX)
CtyL = np.zeros_like(kneeLY)
for i in range(len(kneeLX)):
    CtxL[i], CtyL[i] = COM(hipLX[i],hipLY[i],kneeLX[i],kneeLY[i],dH)
vtLx = np.gradient(CtxL,tNormL)
vtLy = np.gradient(CtyL,tNormL)
atLx = np.gradient(vtLx,tNormL)
atLy = np.gradient(vtLy,tNormL)
omegatL = np.gradient(thighangleL,tNormL)
alphatL = np.gradient(omegatL,tNormL)
dH_xL, dH_yL = calculate_distance_vector(CtxL,CtyL,hipLX,hipLY)
ltdH_xL,ltdH_yL = calculate_distance_vector(CtxL,CtyL,kneeLX,kneeLY)

F_Hx_L = mt*atLx - F_Kx_L
F_Hy_L = mt*atLy + mt*g + F_Ky_L
M_H_l = It*alphatL - dH_xL*F_Hy_L + dH_yL*F_Hx_L + M_K_l - ltdH_yL*F_Kx_L + ltdH_xL*F_Ky_L


## ---------------------------------- ##
## Data for Crouch walking - kinetics ##
## ---------------------------------- ##

data_crouchwalk = np.loadtxt("Project 1/crouch.txt",skiprows=1)
angles_C = np.loadtxt("Project 1/kinematics_crouch_R.txt")
forces_crouch = np.loadtxt("Project 1/crouch_FP.txt",skiprows=1)

## Right gait ##

tOn = 421 # = frame 422 = line 422 - 1 because header
tOff = 526 # = frame 526 = line 526 because header
tCrouchR = data_crouchwalk[tOn:tOff,1]

F_gC_X_tot = forces_crouch[:,11]
F_gC_Y_tot = forces_crouch[:,12]
F_x_Ctot = forces_crouch[:,8]/1000
F_y_Ctot = forces_crouch[:,9]/1000

F_gC_X_s = F_gC_X_tot[::10]
F_gC_Y_s = F_gC_Y_tot[::10]
F_x_Cs = F_x_Ctot[::10]
F_y_Cs = F_y_Ctot[::10]

F_gC_X = F_gC_X_s[tOn:tOff]
F_gC_Y = F_gC_Y_s[tOn:tOff]
F_x_C = F_x_Cs[tOn:tOff]
F_y_C = F_y_Cs[tOn:tOff]

footCX = data_crouchwalk[tOn:tOff,30]/1000
footCY = data_crouchwalk[tOn:tOff,31]/1000
ankleCX = data_crouchwalk[tOn:tOff,24]/1000
ankleCY = data_crouchwalk[tOn:tOff,25]/1000
hipCX = data_crouchwalk[tOn:tOff,18]/1000
hipCY = data_crouchwalk[tOn:tOff,19]/1000
kneeCX = data_crouchwalk[tOn:tOff,21]/1000
kneeCY = data_crouchwalk[tOn:tOff,22]/1000


# Foot equations

footangleC = np.radians(angles_C[:,4])

CfxC = np.zeros_like(ankleCX)
CfyC = np.zeros_like(ankleCY)
for i in range(len(ankleCX)):
    CfxC[i], CfyC[i] = COM(ankleCX[i],ankleCY[i],footCX[i],footCY[i],dA)
vfCx = np.gradient(CfxC,tCrouchR)
vfCy = np.gradient(CfyC,tCrouchR)
afCx = np.gradient(vfCx,tCrouchR)
afCy = np.gradient(vfCy,tCrouchR)
omegafC = np.gradient(footangleC,tCrouchR)
alphafC = np.gradient(omegafC,tCrouchR)
dA_xC, dA_yC = calculate_distance_vector(CfxC,CfyC,ankleCX,ankleCY)
dG_xC,dG_yC = calculate_distance_vector(CfxC,CfyC,F_x_C,F_y_C)

F_Ax_C = mf*afCx - F_gC_X
F_Ay_C = mf*afCy + mf*g - F_gC_Y
M_A_c = If*alphafC - dA_xC*F_Ay_C + dA_yC*F_Ax_C - dG_xC*F_gC_Y + dG_yC*F_gC_X

# Shank equations

shankangleC = np.radians(angles_C[:,3])
CsxC = np.zeros_like(kneeCX)
CsyC = np.zeros_like(kneeCY)
for i in range(len(kneeCX)):
    CsxC[i], CsyC[i] = COM(kneeCX[i],kneeCY[i],ankleCX[i],ankleCY[i],dK)
vsCx = np.gradient(CsxC,tCrouchR)
vsCy = np.gradient(CsyC,tCrouchR)
asCx = np.gradient(vsCx,tCrouchR)
asCy = np.gradient(vsCy,tCrouchR)
omegasC = np.gradient(shankangleC,tCrouchR)
alphasC = np.gradient(omegasC,tCrouchR)
dK_xC, dK_yC = calculate_distance_vector(CsxC,CsyC,kneeCX,kneeCY)
lsdK_xC,lsdK_yC = calculate_distance_vector(CsxC,CsyC,ankleCX,ankleCY)

F_Kx_C = ms*asCx - F_Ax_C
F_Ky_C = ms*asCy + ms*g + F_Ay_C
M_K_c = Is*alphasC - dK_xC*F_Ky_C + dK_yC*F_Kx_C + M_A_c - lsdK_yC*F_Ax_C + lsdK_xC*F_Ay_C

# Thigh equations

thighangleC = np.radians(angles_C[:,2])
CtxC = np.zeros_like(kneeCX)
CtyC = np.zeros_like(kneeCY)
for i in range(len(kneeCX)):
    CtxC[i], CtyC[i] = COM(hipCX[i],hipCY[i],kneeCX[i],kneeCY[i],dH)
vtCx = np.gradient(CtxC,tCrouchR)
vtCy = np.gradient(CtyC,tCrouchR)
atCx = np.gradient(vtCx,tCrouchR)
atCy = np.gradient(vtCy,tCrouchR)
omegatC = np.gradient(thighangleC,tCrouchR)
alphatC = np.gradient(omegatC,tCrouchR)
dH_xC, dH_yC = calculate_distance_vector(CtxC,CtyC,hipCX,hipCY)
ltdH_xC,ltdH_yC = calculate_distance_vector(CtxC,CtyC,kneeCX,kneeCY)

F_Hx_C = mt*atCx - F_Kx_C
F_Hy_C = mt*atCy + mt*g + F_Ky_C
M_H_c = It*alphatC - dH_xC*F_Hy_C + dH_yC*F_Hx_C + M_K_c - ltdH_yC*F_Kx_C + ltdH_xC*F_Ky_C

## Graphs ##
## ------ ##

# Ankle angle comparison #

fig40, ax40 = plt.subplots()
fig40.set_size_inches(15, 8)
 
ax40.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,-M_A_r/weight,c='mediumblue',label='Right Gait')
ax40.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
ax40.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,-M_A_l/weight, c='darkorange',label = 'Left Gait')
ax40.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')
ax40.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,-M_A_c/weight, c='darkcyan',label = 'Crouch Gait')
ax40.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkcyan',linestyle='dotted',label='toe-off crouch gait')

plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
plt.ylabel("Ankle moment [Nm/kg]",fontsize=20)
plt.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

plt.text(-11, -0.4, 'Dorsiflexion', color='red',fontsize=18, rotation=90)
plt.text(-11, 1.3, 'Plantarflexion', color='green',fontsize=18, rotation=90)
plt.axhline(y=0, color='k')

ax40.vlines(x=0.2, ymin=0, ymax=45, color='green',linewidth=12)
ax40.vlines(x=0.2, ymin=-115, ymax=0, color='red',linewidth=12)

plt.axis([0,100,-0.4,1.8])
plt.legend(fontsize= 20)

# Knee angle comparison #


#Moment

fig3, ax3 = plt.subplots()
fig3.set_size_inches(15, 8)
 
ax3.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,M_K_r/weight,c='mediumblue',label='Right Gait')
ax3.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
ax3.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,M_K_l/weight, c='darkorange',label = 'Left Gait')
ax3.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')
ax3.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,M_K_c/weight, c='darkcyan',label = 'Crouch Gait')
ax3.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkcyan',linestyle='dotted',label='toe-off crouch gait')

plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
plt.ylabel("Knee moment  [Nm/kg]",fontsize=20)
plt.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

plt.text(-11, -0.5, 'Flexion', color='red',fontsize=18, rotation=90)
plt.text(-11, 0.5, 'Hyperextension', color='green',fontsize=18, rotation=90)
plt.axhline(y=0, color='k')

ax3.vlines(x=0.2, ymin=0, ymax=55, color='green',linewidth=12)
ax3.vlines(x=0.2, ymin=-5, ymax=0, color='red',linewidth=12)

plt.axis([0,100,-0.5,1])
plt.legend(fontsize= 20)

# Hip angle comparison #

fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 8)
 
ax2.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,-M_H_r/weight,c='mediumblue',label='Right Gait')
ax2.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off right gait')
ax2.plot((tNormL-tNormL[0])/(tNormL[-1]-tNormL[0])*100,-M_H_l/weight, c='darkorange',label = 'Left Gait')
ax2.axvline((tNormL[316-tOnL-1]-tNormL[0])/(tNormL[-1]-tNormL[0])*100,color='darkorange',linestyle='dotted',label='toe-off left gait')
ax2.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,-M_H_c/weight, c='darkcyan',label = 'Crouch Gait')
ax2.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkcyan',linestyle='dotted',label='toe-off crouch gait')

plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
plt.ylabel("Hip moment [Nm/kg]",fontsize=20)
plt.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

plt.text(-11, -2.5, 'Flexion', color='red',fontsize=18, rotation=90)
plt.text(-11, 2, 'Extension', color='green',fontsize=18, rotation=90)
plt.axhline(y=0, color='k')

ax2.vlines(x=0.2, ymin=0, ymax=40, color='green',linewidth=12)
ax2.vlines(x=0.2, ymin=-15, ymax=0, color='red',linewidth=12)

plt.axis([0,100,-2.5,3.0])
plt.legend(fontsize= 20)

plt.show()

