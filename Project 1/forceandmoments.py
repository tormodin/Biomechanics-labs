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
# ...

# Step 1: Calculate Joint Centers
# ...

# Step 2: Calculate Segment Positions
# ...

# Step 3: Calculate Linear Accelerations
# ...

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
