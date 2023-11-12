import numpy as np
import matplotlib.pyplot as plt
from kinematics import *

# Load kinematic and force plate data


force_plate_data = np.loadtxt("Project 1/walking_FP.txt", skiprows=1) 
kinematic_data = np.loadtxt("Project 1/walking.txt", skiprows=1)
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


# Step 1: Calculate Joint Centers------- #We arleady have that 
# Step 2: Calculate Segment Positions------- # I think we can skip this because it just renames the variables
# Step 3: Calculate Linear Accelerations------- # Be carefull to take the two time steps before the inital frame because we derivate twice
# Assuming time steps are constant
time_index = 1
kinematics_time_column = kinematic_data[:, time_index]
dt = kinematics_time_column[1] - kinematics_time_column[0] #what are u using for dt?

# Pelvis linear accelerations
pelvis_linear_acceleration_y = np.gradient(np.gradient(pelpRX, dt), dt)
pelvis_linear_acceleration_z = np.gradient(np.gradient(pelpRY, dt), dt) #we are using z and x right?
# Hip linear accelerations
hip_linear_acceleration_y = np.gradient(np.gradient(hipRY, dt), dt)
hip_linear_acceleration_z = np.gradient(np.gradient(hipRX, dt), dt)

# Knee linear accelerations
knee_linear_acceleration_y = np.gradient(np.gradient(kneeRY, dt), dt)
knee_linear_acceleration_z = np.gradient(np.gradient(kneeRX, dt), dt)

# Ankle linear accelerations
ankle_linear_acceleration_y = np.gradient(np.gradient(ankleRY, dt), dt)
ankle_linear_acceleration_z = np.gradient(np.gradient(ankleRX, dt), dt)


# Step 4: Calculate Angular Accelerations------- #not sure about the data

# Hip angular accelerations
hip_angular_acceleration_y = np.gradient(np.gradient(angle_degrees_hip, dt), dt)
hip_angular_acceleration_z = np.gradient(np.gradient(angle_degrees_hip, dt), dt)

# Knee angular accelerations
knee_angular_acceleration_y = np.gradient(np.gradient(kneeangleR, dt), dt)
knee_angular_acceleration_z = np.gradient(np.gradient(kneeangleR, dt), dt)

# Ankle angular accelerations
ankle_angular_acceleration_y = np.gradient(np.gradient(ankleangleR, dt), dt)
ankle_angular_acceleration_z = np.gradient(np.gradient(ankleangleR, dt), dt)


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
