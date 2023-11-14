import numpy as np
import matplotlib.pyplot as plt

# Load kinematic and force plate data

from kinematics import *
#kinematic_data = np.loadtxt("Project 1/walking.txt", skiprows=1) #You can take this from the code kinematics----
#force_plate_data = np.loadtxt("Project 1/walking_FP.txt", skiprows=1) 

# Assuming the file "walking.txt" is in the "Project 1" folder relative to your current working directory.
file_path = "Project 1/walking.txt"
kinematic_data = np.loadtxt(file_path, skiprows=1)
# Assuming the file "walking_FP.txt" is in the "Project 1" folder relative to your current working directory.
file_path = "Project 1/walking_FP.txt"
force_plate_data = np.loadtxt(file_path, skiprows=1)

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

thigh_mass = weight * (thigh_mass_percentage / 100)
shank_mass = weight * (shank_mass_percentage / 100)
foot_mass = weight * (foot_mass_percentage / 100)
pelvis_mass = weight * pelvis_mass_ratio

thigh_COM = thigh_length * (thigh_com_percentage / 100)
shank_COM = shank_length * (shank_com_percentage / 100)
foot_COM = foot_length * (foot_com_percentage / 100)
pelvis_COM = height * pelvis_COM_ratio

hip_mass = thigh_mass + pelvis_mass
knee_mass = shank_mass
ankle_mass = foot_mass
#Until here I think it is great



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
kinematics_time_column = kinematic_data[:, time_index] #Take frame of gait like in line 16 to 22 of kinematics
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


# Step 1: Calculate Joint Centers------- #We arleady have that 
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

#angles -> All these angles are already in kinematics
trunk_angle = np.degrees(np.arctan2(trunk_y_column - pelvis_y_column, trunk_z_column - pelvis_z_column))


pelvis_angle = np.degrees(np.arctan2(pelvis_y_column - pelvis_y_column, pelvis_z_column - pelvis_z_column))

hip_angle = np.degrees(np.arctan2(hip_y_column - pelvis_y_column, hip_z_column - pelvis_z_column))

knee_angle = np.degrees(np.arctan2(knee_y_column - hip_y_column, knee_z_column - hip_z_column))

ankle_angle = np.degrees(np.arctan2(ankle_y_column - knee_y_column, ankle_z_column - knee_z_column))

# Step 2: Calculate Segment Positions------- # I think we can skip this because it just renames the variables
# Assuming symmetric placement of joints, let's calculate the segment positions in the sagittal plane

# Pelvis position
pelvis_position_y = pelvis_y_column
pelvis_position_z = pelvis_z_column

# Hip position
hip_position_y = hip_y_column
hip_position_z = hip_z_column

# Knee position
knee_position_y = knee_y_column
knee_position_z = knee_z_column

# Ankle position
ankle_position_y = ankle_y_column
ankle_position_z = ankle_z_column


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


# Step 3: Calculate Linear Accelerations------- # Be carefull to take the two time steps before the inital frame because we derivate twice
# Assuming time steps are constant
dt = kinematics_time_column[1] - kinematics_time_column[0]

# Pelvis linear accelerations
pelvis_linear_acceleration_y = np.gradient(np.gradient(pelvis_position_y, dt), dt)
pelvis_linear_acceleration_z = np.gradient(np.gradient(pelvis_position_z, dt), dt)
# Hip linear accelerations
hip_linear_acceleration_y = np.gradient(np.gradient(hip_position_y, dt), dt)
hip_linear_acceleration_z = np.gradient(np.gradient(hip_position_z, dt), dt)

# Knee linear accelerations
knee_linear_acceleration_y = np.gradient(np.gradient(knee_position_y, dt), dt)
knee_linear_acceleration_z = np.gradient(np.gradient(knee_position_z, dt), dt)

# Ankle linear accelerations
ankle_linear_acceleration_y = np.gradient(np.gradient(ankle_position_y, dt), dt)
ankle_linear_acceleration_z = np.gradient(np.gradient(ankle_position_z, dt), dt)


# Step 4: Calculate Angular Accelerations-------

# Hip angular accelerations
hip_angular_acceleration_y = np.gradient(np.gradient(hip_angle, dt), dt)
hip_angular_acceleration_z = np.gradient(np.gradient(hip_angle, dt), dt)

# Knee angular accelerations
knee_angular_acceleration_y = np.gradient(np.gradient(knee_angle, dt), dt)
knee_angular_acceleration_z = np.gradient(np.gradient(knee_angle, dt), dt)

# Ankle angular accelerations
ankle_angular_acceleration_y = np.gradient(np.gradient(ankle_angle, dt), dt)
ankle_angular_acceleration_z = np.gradient(np.gradient(ankle_angle, dt), dt)

#First compute the Center of mass of each segment using the positions of its ends (in file walking.txt) + angles of each segment (in code kinematics.txt)

# Step 5: Calculate Net Joint Forces and Moments------- #Rewriting the equations we did in course 4
# Hip position
hip_position_y = hip_y_column
hip_position_z = hip_z_column

# Net Joint Forces
hip_force_y = hip_mass * hip_linear_acceleration_y
hip_force_z = hip_mass * hip_linear_acceleration_z

# Net Joint Moments
hip_moment_y = hip_mass * hip_angular_acceleration_y
hip_moment_z = hip_mass * hip_angular_acceleration_z

# Knee
knee_mass = shank_mass

# Net Joint Forces
knee_force_y = knee_mass * knee_linear_acceleration_y
knee_force_z = knee_mass * knee_linear_acceleration_z

# Net Joint Moments
knee_moment_y = knee_mass * knee_angular_acceleration_y
knee_moment_z = knee_mass * knee_angular_acceleration_z

# Ankle
ankle_mass = foot_mass

# Net Joint Forces
ankle_force_y = ankle_mass * ankle_linear_acceleration_y
ankle_force_z = ankle_mass * ankle_linear_acceleration_z

# Net Joint Moments
ankle_moment_y = ankle_mass * ankle_angular_acceleration_y
ankle_moment_z = ankle_mass * ankle_angular_acceleration_z

# Step 5: Calculate Net Joint Forces and Moments-------
# Assuming symmetric placement of joints, let's calculate for the hip joint in the sagittal plane

# Constants
g = 9.81  # gravitational acceleration in m/s^2

# Hip
hip_mass = thigh_mass_percentage / 100 * weight
hip_com_percentage = thigh_com_percentage / 100
print("Type of hip_linear_acceleration_y:", type(hip_linear_acceleration_y))


print("Shapes:")
#print("hip_mass * hip_linear_acceleration_y:", hip_mass * hip_linear_acceleration_y.shape)
print("force_plate_data[:, 4]:", force_plate_data[:, 4].shape)
print("force_plate_data[:, 10]:", force_plate_data[:, 10].shape)



# Net Joint Forces
hip_force_y = hip_mass * hip_linear_acceleration_y + force_plate_data[:, 4][:, np.newaxis] + force_plate_data[:, 10][:, np.newaxis]
hip_force_z = hip_mass * hip_linear_acceleration_z + force_plate_data[:, 5][:, np.newaxis] + force_plate_data[:, 11][:, np.newaxis]

hip_lever_arm_y = hip_position_y - (force_plate_data[:, 1][:, np.newaxis] + force_plate_data[:, 7][:, np.newaxis]) / 2
hip_lever_arm_z = hip_position_z - (force_plate_data[:, 2][:, np.newaxis] + force_plate_data[:, 8][:, np.newaxis]) / 2

# Net Joint Moments
hip_moment_y = hip_force_z * hip_lever_arm_z - hip_force_y * hip_lever_arm_y

# Convert forces and moments to N and Nm
hip_force_y_N = hip_force_y / 1000  # Convert to N
hip_force_z_N = hip_force_z / 1000
hip_moment_y_Nm = hip_moment_y / 1000  # Convert to Nm

# Display the results
print("Hip Net Joint Forces (N):")
print("Y-axis:", hip_force_y_N)
print("Z-axis:", hip_force_z_N)

print("\nHip Net Joint Moments (Nm):")
print("Y-axis:", hip_moment_y_Nm)

# Convert forces and moments to N and Nm for Knee
knee_force_y_N = knee_force_y / 1000  # Convert to N
knee_force_z_N = knee_force_z / 1000
knee_moment_y_Nm = knee_moment_y / 1000  # Convert to Nm

# Display the results for Knee
print("\nKnee Net Joint Forces (N):")
print("Y-axis:", knee_force_y_N)
print("Z-axis:", knee_force_z_N)

print("\nKnee Net Joint Moments (Nm):")
print("Y-axis:", knee_moment_y_Nm)


# Convert forces and moments to N and Nm for Ankle
ankle_force_y_N = ankle_force_y / 1000  # Convert to N
ankle_force_z_N = ankle_force_z / 1000
ankle_moment_y_Nm = ankle_moment_y / 1000  # Convert to Nm

# Display the results for Ankle
print("\nAnkle Net Joint Forces (N):")
print("Y-axis:", ankle_force_y_N)
print("Z-axis:", ankle_force_z_N)

print("\nAnkle Net Joint Moments (Nm):")
print("Y-axis:", ankle_moment_y_Nm)


# Step 6: Distribute Net Joint Forces-------
# Step 6: Distribute Net Joint Forces-------
# Assuming you have information about muscle moment arms and activation levels
# You might need to replace the placeholders with actual values based on your model

# Placeholder values, replace with actual data
hip_abductor_moment_arm = 0.1  # Example moment arm for the hip abductors
hip_extensor_moment_arm = 0.2  # Example moment arm for the hip extensors
knee_flexor_moment_arm = 0.15  # Example moment arm for the knee flexors
knee_extensor_moment_arm = 0.25  # Example moment arm for the knee extensors
ankle_dorsiflexor_moment_arm = 0.1  # Example moment arm for the ankle dorsiflexors
ankle_plantarflexor_moment_arm = 0.2  # Example moment arm for the ankle plantarflexors


# Distribute net joint forces to muscles based on moment arms
hip_abductor_force = hip_force_y_N / hip_abductor_moment_arm
hip_extensor_force = hip_force_y_N / hip_extensor_moment_arm


# Distribute net joint forces to muscles based on moment arms
knee_flexor_force = knee_force_y_N / knee_flexor_moment_arm
knee_extensor_force = knee_force_y_N / knee_extensor_moment_arm

# Distribute net joint forces to muscles based on moment arms
ankle_dorsiflexor_force = ankle_force_y_N / ankle_dorsiflexor_moment_arm
ankle_plantarflexor_force = ankle_force_y_N / ankle_plantarflexor_moment_arm


# Display the distributed forces
print("\nDistributed Muscle Forces (N):")
print("Hip Abductor Force:", hip_abductor_force)
print("Hip Extensor Force:", hip_extensor_force)

# Display the distributed forces
print("\nDistributed Muscle Forces at Knee (N):")
print("Knee Flexor Force:", knee_flexor_force)
print("Knee Extensor Force:", knee_extensor_force)

# Display the distributed forces
print("\nDistributed Muscle Forces at Ankle (N):")
print("Ankle Dorsiflexor Force:", ankle_dorsiflexor_force)
print("Ankle Plantarflexor Force:", ankle_plantarflexor_force)




# Step 7: Calculate Muscle Forces-------
# ...

# Step 8: Calculate Joint Reaction Forces
# ...

# Step 9: Calculate Joint Moments
# ...

# Step 10: Calculate Joint Power
# Plotting knee moments over time
fig, ax = plt.subplots(figsize=(15, 8))

# Plot knee moment in the y-axis
ax.plot(kinematics_time_column, knee_moment_y_Nm, label='Knee Moment (Nm)', color='green')

# Customize your plot
ax.legend()
ax.set_xlabel("Time [s]", fontsize=16)
ax.set_ylabel("Knee Moment [Nm]", fontsize=16)
ax.set_title("Knee Moment Over Time", fontsize=18)
ax.grid(True)

# Show the plot
plt.show()


# Step 11: Normalize Data if needed
# ...
""""
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
"""