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
angles_C = np.loadtxt("Project 1/kinematics_crouch_R.txt")
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


# Foot equations

footangleR = angles_normR[:,4]

Cfx = np.zeros_like(ankleRX)
Cfy = np.zeros_like(ankleRY)
for i in range(len(ankleRX)):
    Cfx[i], Cfy[i] = COM(ankleRX[i],ankleRY[i],footRX[i],footRY[i],dA)
vfRx = np.gradient(Cfx,tNormR)
vfRy = np.gradient(Cfy,tNormR)
afRx = np.gradient(vfRx,tNormR)
afRy = np.gradient(vfRy,tNormR)
omegafR = np.gradient(footangleR,tNormR)
alphafR = np.gradient(omegafR,tNormR)
dA_x, dA_y = calculate_distance_vector(Cfx,Cfy,ankleRX,ankleRY)
dG_x,dG_y = calculate_distance_vector(Cfx,Cfy,F_x_R,F_y_R)

F_Ax_R = mf*afRx - F_gR_X
F_Ay_R = mf*afRy + mf*g - F_gR_Y
M_A_r = If*alphafR - dA_x*F_Ay_R + dA_y*F_Ax_R - dG_x*F_gR_Y + dG_y*F_gR_X

print(M_A_r)

# Ankle angle comparison #

fig40, ax40 = plt.subplots()
fig40.set_size_inches(15, 8)
 
ax40.plot((tNormR-tNormR[0])/(tNormR[-1]-tNormR[0])*100,M_A_r,c='mediumblue',label='Normal Gait')
# ax40.axvline((tNormR[270-tOnR-1]-tNormR[0])/(tNormR[-1]-tNormR[0])*100,color='mediumblue',linestyle='dotted',label='toe-off normal gait')
# ax40.plot((tCrouchR-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,ankleangleC, c='darkorange',label = 'Crouch Gait')
# ax40.axvline((tCrouchR[484-tOn-1]-tCrouchR[0])/(tCrouchR[-1]-tCrouchR[0])*100,color='darkorange',linestyle='dotted',label='toe-off crouch gait')

plt.xlabel("Percentage of gait cycle [%]",fontsize=20)
plt.ylabel("Ankle moment [Nm]",fontsize=20)
plt.grid('True')

for tickLabel in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
  tickLabel.set_fontsize(16)

plt.text(-11, -110, 'Plantarflexion', color='red',fontsize=18, rotation=90)
plt.text(-11, 5, 'Dorsiflexion', color='green',fontsize=18, rotation=90)
plt.axhline(y=0, color='k')

ax40.vlines(x=0.2, ymin=0, ymax=45, color='green',linewidth=12)
ax40.vlines(x=0.2, ymin=-115, ymax=0, color='red',linewidth=12)

#plt.axis([0,100,-115,45])
plt.legend(fontsize= 20)

plt.show()

