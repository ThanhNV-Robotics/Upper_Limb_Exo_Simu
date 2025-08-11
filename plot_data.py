import numpy as np
import matplotlib.pyplot as plt


loaded_data = np.load('test_sin_data.npy')
data = np.array(loaded_data)  # Convert back to NumPy array
t = data[:, 0]                  # time
theta_1_ref = np.rad2deg(data[:, 1])
theta_2_ref = np.rad2deg(data[:, 2])
theta_3_ref = np.rad2deg(data[:, 3])

theta_1_fb = np.rad2deg(data[:, 4])
theta_2_fb = np.rad2deg(data[:, 5])
theta_3_fb = np.rad2deg(data[:, 6])

torque_joint_1 = data[:,7]
torque_joint_2 = data[:,8]
torque_joint_3 = data[:,9]

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})  # sets default font size to 14 for everything

plt.plot(t, theta_1_ref, label='ref_shoulder_flexion', color='blue', linestyle='--')
plt.plot(t, theta_2_ref, label='ref_shoulder_inner_rot', color='black', linestyle='--')
plt.plot(t, theta_3_ref, label='ref_elbow', color='red', linestyle='--')

plt.plot(t, theta_1_fb, label='fb_shoulder_flexion', color='blue', linestyle='-')
plt.plot(t, theta_2_fb, label='fb_shoulder_inner_rot', color='black', linestyle='-')
plt.plot(t, theta_3_fb, label='fb_elbow', color='red', linestyle='-')

plt.xlabel('Time (s)')
plt.ylabel('angle (deg)')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})  # sets default font size to 14 for everything

plt.plot(t, torque_joint_1, label='torque_shoulder_flex', color='blue', linestyle='-')
plt.plot(t, torque_joint_2, label='torque_shoulder_inner_rot', color='black', linestyle='-')
plt.plot(t, torque_joint_3, label='torque_elbow', color='red', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('torque (Nm)')
plt.legend()
plt.grid(True)

plt.show()