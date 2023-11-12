import numpy as np
import matplotlib.pyplot as plt

# Load kinematic and force plate data

from kinematics.py import *
force_plate_data = np.loadtxt("Project 1/walking_FP.txt", skiprows=1) 
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
