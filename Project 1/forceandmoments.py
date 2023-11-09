import numpy as np
import matplotlib.pyplot as plt

# Load kinematic and force plate data

kinematic_data = np.loadtxt("walking.txt", skiprows=1)
force_plate_data = np.loadtxt("walking_FP.txt", skiprows=1)

# Constants and anthropometric data
height = 1680  # in mm
weight = 71.5  # in kg

# Anthropometric data for thigh
thigh_length = 400  # in mm
thigh_mass_percentage = 10  # Replace with actual percentage
thigh_com_percentage = 40  # Replace with actual percentage

# Anthropometric data for shank
shank_length = 400  # in mm
shank_mass_percentage = 5  # Replace with actual percentage
shank_com_percentage = 40  # Replace with actual percentage

# Anthropometric data for foot
foot_length = 200  # in mm
foot_mass_percentage = 1  # Replace with actual percentage
foot_com_percentage = 50  # Replace with actual percentage

pelvis_mass_ratio = 0.142
pelvis_COM_ratio = 0.25


# Extract relevant columns from your data
# Assuming your data variable is a NumPy array with appropriate column indices

# Replace these indices with the actual column indices in your data
time_index = 1
pelvis_y_index = 4
trunk_y_index = 7
hip_y_index = 16
knee_y_index = 19
ankle_y_index = 22

# Extract relevant columns from your data
kinematics_time_column = kinematic_data[:, time_index]
pelvis_y_column = kinematic_data[:, pelvis_y_index]
trunk_y_column = kinematic_data[:, trunk_y_index]
hip_y_column = kinematic_data[:, hip_y_index]
knee_y_column = kinematic_data[:, knee_y_index]
ankle_y_column = kinematic_data[:, ankle_y_index]

# Now you have the y-coordinates for each joint in the sagittal plane
# Assuming your data variable is a NumPy array with appropriate column indices

# Replace these indices with the actual column indices in your data
pelvis_z_index = 5
trunk_z_index = 8
hip_z_index = 17
knee_z_index = 20
ankle_z_index = 23

# Extract relevant columns from your data
pelvis_z_column = kinematic_data[:, pelvis_z_index]
trunk_z_column = kinematic_data[:, trunk_z_index]
hip_z_column = kinematic_data[:, hip_z_index]
knee_z_column = kinematic_data[:, knee_z_index]
ankle_z_column = kinematic_data[:, ankle_z_index]

# Now you have the z-coordinates for each joint in the sagittal plane


# Step 1: Calculate Joint Centers
#hip
RHJC_y = (pelvis_y_column + hip_y_column) / 2
RHJC_z = (pelvis_z_column + hip_z_column) / 2

LHJC_y = (pelvis_y_column + hip_y_column) / 2  # Assuming symmetric placement
LHJC_z = (pelvis_z_column + hip_z_column) / 2

#knee
RKJC_y = (hip_y_column + knee_y_column) / 2
RKJC_z = (hip_z_column + knee_z_column) / 2

LKJC_y = (hip_y_column + knee_y_column) / 2  # Assuming symmetric placement
LKJC_z = (hip_z_column + knee_z_column) / 2
#ankle
RAJC_y = (knee_y_column + ankle_y_column) / 2
RAJC_z = (knee_z_column + ankle_z_column) / 2

LAJC_y = (knee_y_column + ankle_y_column) / 2  # Assuming symmetric placement
LAJC_z = (knee_z_column + ankle_z_column) / 2

#angles
trunk_angle = np.degrees(np.arctan2(trunk_y_column - pelvis_y_column, trunk_z_column - pelvis_z_column))


pelvis_angle = np.degrees(np.arctan2(pelvis_y_column - pelvis_y_column, pelvis_z_column - pelvis_z_column))

hip_angle = np.degrees(np.arctan2(hip_y_column - pelvis_y_column, hip_z_column - pelvis_z_column))

knee_angle = np.degrees(np.arctan2(knee_y_column - hip_y_column, knee_z_column - hip_z_column))

ankle_angle = np.degrees(np.arctan2(ankle_y_column - knee_y_column, ankle_z_column - knee_z_column))

# Step 2: Calculate Segment Positions
# Assuming symmetric placement of joints, let's calculate the segment positions in the sagittal plane

# Pelvis position
pelvis_position_y = pelvis_y_column
pelvis_position_z = pelvis_z_column

# Trunk position (assumed to be halfway between pelvis and thorax markers)
trunk_position_y = (pelvis_y_column + trunk_y_column) / 2
trunk_position_z = (pelvis_z_column + trunk_z_column) / 2

# Right Hip position
RHJC_position_y = RHJC_y
RHJC_position_z = RHJC_z

# Left Hip position
LHJC_position_y = LHJC_y
LHJC_position_z = LHJC_z

# Right Knee position
RKJC_position_y = RKJC_y
RKJC_position_z = RKJC_z

# Left Knee position
LKJC_position_y = LKJC_y
LKJC_position_z = LKJC_z

# Right Ankle position
RAJC_position_y = RAJC_y
RAJC_position_z = RAJC_z

# Left Ankle position
LAJC_position_y = LAJC_y
LAJC_position_z = LAJC_z


# Step 3: Calculate Linear Accelerations
# Assuming time steps are constant
dt = kinematics_time_column[1] - kinematics_time_column[0]

# Pelvis linear accelerations
pelvis_linear_acceleration_y = np.gradient(np.gradient(pelvis_position_y, dt), dt)
pelvis_linear_acceleration_z = np.gradient(np.gradient(pelvis_position_z, dt), dt)

# Repeat the above process for other joint centers and segment endpoints as needed

# Step 4: Calculate Angular Accelerations
# ...

# Step 5: Calculate Net Joint Forces and Moments
# ...

# Step 6: Distribute Net Joint Forces
# ...

# Step 7: Calculate Muscle Forces
# ...

# Step 8: Calculate Joint Reaction Forces
# ...

# Step 9: Calculate Joint Moments
# ...

# Step 10: Calculate Joint Power
# ...

# Step 11: Normalize Data if needed
# ...

# Plotting (you can add more plots as needed)
fig, ax = plt.subplots(figsize=(15, 8))

# Plot joint angles or other relevant kinematics
ax.plot(data[:, 1], data[:, 33], label='Trunk Angle', color='blue')

# Add more plots for joint angles, moments, powers, etc.

# Customize your plot
ax.legend()
ax.set_xlabel("Time [s]", fontsize=16)
ax.set_ylabel("Angle/Force/Moment", fontsize=16)
ax.grid(True)

plt.show()
