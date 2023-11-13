import numpy as np
import matplotlib.pyplot as plt
from kinematics import *

# Load kinematic and force plate data


force_plate_data = np.loadtxt("Project 1/walking_FP.txt", skiprows=1) 
kinematic_data = np.loadtxt("Project 1/walking.txt", skiprows=1)


# Constants and anthropometric data
height = 1680  # in mm
weight = 71.5  # in kg


#have to redo
hip_y_index = 16
hip_y_column = kinematic_data[:, hip_y_index]

hip_z_index = 17
hip_z_column = kinematic_data[:, hip_z_index]

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
hip_angular_acceleration_y = np.gradient(np.gradient(hipangleR, dt), dt)
hip_angular_acceleration_z = np.gradient(np.gradient(hipangleR, dt), dt)

# Knee angular accelerations
knee_angular_acceleration_y = np.gradient(np.gradient(kneeangleR, dt), dt)
knee_angular_acceleration_z = np.gradient(np.gradient(kneeangleR, dt), dt)

# Ankle angular accelerations
ankle_angular_acceleration_y = np.gradient(np.gradient(ankleangleR, dt), dt)
ankle_angular_acceleration_z = np.gradient(np.gradient(ankleangleR, dt), dt)

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

# Step 6: Calculate Net Joint Forces and Moments-------
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

# Step 7: Distribute Net Joint Forces-------
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

"""
# Display the distributed forces
print("\nDistributed Muscle Forces (N):")
print("Hip Abductor Force:", hip_abductor_force)
print("Hip Extensor Force:", hip_extensor_force)

# Display the distributed forces
print("\nDistributed Muscle Forces at Knee (N):")
print("Knee Flexor Force:", knee_flexor_force)
print("Knee Extensor Force:", knee_extensor_force)
"""
# Display the distributed forces
print("\nDistributed Muscle Forces at Ankle (N):")
print("Ankle Dorsiflexor Force:", ankle_dorsiflexor_force)
print("Ankle Plantarflexor Force:", ankle_plantarflexor_force)
#code to give correct variable names
# Moment on the Foot
M_foot = dAx * FAy - dAy * FAx + dGR * FGR

# Moment on the Shank
FAx = mjax - FGRx
FAy = mjay + mfg - FGRy
MA_shank = IfG * g + dAx * FAy + dAy * FAx - dGR * FGR

# Moment on the Ankle
MA_ankle = msax - FAx
FKy = msay + FAy + msg
HK_ankle = ISG_s - dkx * FKy + dky * FAx + MA_ankle - (ls - dk) * FAy + (ls - dks) * FAx

# Moment on the Thigh
FHx = mTax + Fkx
FHy = mTay + mTg + FKy
HH_thigh = ITG_T - dHx * FHy + dHy * FHx + HK_ankle + (lT - dH) * FKy - (lT - dH) * Fkx




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
