import mujoco
import numpy as np
from mujoco.viewer import launch_passive

class PD_Impedence_Control: # MIMO, joint position PD impedence controller
    def __init__(self, Kp, Kd):
        n = np.shape(Kp)[1]
        self.Kp = Kp
        self.Kd = Kd
        
    def PD_Control_Calculate (self, ref_pos, ref_vel, fb_pos, fb_vel): # MIMO
        # can add feedforward torque if needed

        torque = self.Kp@(ref_pos - fb_pos) + self.Kd@(ref_vel - fb_vel)
        return torque # return torque vector applied to robot's joints

def Get_Joints_Pos (mj_model, mj_data):
    q = []
    joint_names = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    for joint in joint_names:
        q.append(mj_data.joint(joint).qpos)
    n = len(q)
    if n> 12:
        q = q[1:] # ignore the position and orientation of the torso, only take the joint position

    return np.array(q).reshape(-1)

def Get_Joint_Vel (mj_model, mj_data):
    q_vel = []
    for i in range(mj_model.nv):
        q_vel.append(mj_data.qvel[i])

    n = len(q_vel)
    if n> 12:
        q_vel = q_vel[1:] # ignore the velocity of the torso, only take the joint position
    return np.array(q_vel).reshape(-1)

if __name__ == "__main__":
    starting_time = 0.5 # sec, starting time of the moving function
    # ──────────────────────────────────────────────────────────────────────
    #   Load xml mujoco model
    # ──────────────────────────────────────────────────────────────────────
    mujoco_xml_path = 'mjcf/upper_limb_exo.xml'
    mj_model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
    mj_data = mujoco.MjData(mj_model)
    time_step = mj_model.opt.timestep # time step of mujoco simulator
    n_dofs = mj_model.nv # total dofs, =3 in this case

    # ──────────────────────────────────────────────────────────────────────
    #   Init Low-level PD impedence controller
    # ──────────────────────────────────────────────────────────────────────
    Joint_Kp = np.diag([300, 100, 300]) # N.m/rad
    Joint_Kd = np.diag([1.25, 1.25, 1.25]) # N.ms/rad
    PDController= PD_Impedence_Control(Joint_Kp, Joint_Kd) # init robot's joint PD controller

    # ──────────────────────────────────────────────────────────────────────
    #   Init states and observations
    # ──────────────────────────────────────────────────────────────────────
    theta_1_amp_ref = np.deg2rad(110) # amplitude for sine ref signal
    theta_1_freq_ref = 0.5 #Hz, sine frequency

    theta_2_amp_ref = np.deg2rad(30) # amplitude for sine ref signal
    theta_2_freq_ref = 0.5 #Hz, sine frequency

    theta_3_amp_ref = np.deg2rad(110) # amplitude for sine ref signal
    theta_3_freq_ref = 0.5 #Hz, sine frequency

    theta_1_ref = 0
    theta_2_ref = 0
    theta_3_ref = 0

    theta_1_vel_ref = 0
    theta_2_vel_ref = 0
    theta_3_vel_ref = 0

    robot_joint_ref_angle = np.array([theta_1_ref,theta_2_ref,theta_3_ref])
    robot_joint_ref_velocity = np.array([theta_1_vel_ref, theta_2_vel_ref, theta_3_vel_ref])
    robot_joint_feb_angl = np.zeros(n_dofs) # feedback joints' position
    robot_joint_feb_vel = np.zeros(n_dofs)# feedback joints' velocity
    save_data = [] # to save and plot simulation data
    # ──────────────────────────────────────────────────────────────────────
    #   control system's parameters
    # ──────────────────────────────────────────────────────────────────────
    f_control = 500 # Hz controller frequency
    dt_control = 1/f_control # # time step for each control loop
    t_control = 0
    time_count = 0

    viewer = launch_passive(mj_model, mj_data)

    while viewer.is_running():
    #while mj_data.time <= 5:
        
        mujoco.mj_step(mj_model, mj_data) # forward dynamics, perform a one-time-step simulation
        simu_time = mj_data.time # simulation time 

        # ──────────────────────────────────────────────────────────────────────
        #   Get feedback joints' angles and velocity
        # ──────────────────────────────────────────────────────────────────────
        robot_joint_feb_angl = Get_Joints_Pos(mj_model, mj_data) # Get feedback position
        robot_joint_feb_vel = Get_Joint_Vel(mj_model, mj_data)

        if (simu_time >= starting_time): # start ZMP_preview control 
            time_count += 1

            if time_count >= (int)(dt_control/time_step):
                t_control += dt_control
                # perform the main control loop in here
                theta_1_ref = 0.5*theta_1_amp_ref*np.cos(2*np.pi*theta_1_freq_ref*t_control) - 0.5*theta_1_amp_ref                 
                theta_1_vel_ref = -0.5*theta_1_amp_ref*2*np.pi*theta_1_freq_ref*np.sin(2*np.pi*theta_1_freq_ref*t_control)

                theta_2_ref = theta_2_amp_ref*np.sin(2*np.pi*theta_2_freq_ref*t_control)
                theta_2_vel_ref = theta_2_amp_ref*2*np.pi*theta_2_freq_ref*np.cos(2*np.pi*theta_2_freq_ref*t_control)

                theta_3_ref = 0.5*theta_3_amp_ref*np.cos(2*np.pi*theta_3_freq_ref*t_control) - 0.5*theta_3_amp_ref                 
                theta_3_vel_ref = -0.5*theta_3_amp_ref*2*np.pi*theta_3_freq_ref*np.sin(2*np.pi*theta_3_freq_ref*t_control)



                robot_joint_ref_angle = np.array([theta_1_ref,theta_2_ref,theta_3_ref]).reshape(-1)
                robot_joint_ref_velocity = np.array([theta_1_vel_ref, theta_2_vel_ref, theta_3_vel_ref]).reshape(-1)

                time_count = 0       

        else: #       
            # do nothing
            pass
            
        # Joint Position Control Layer: Use conventional PD controller        
        Joint_Tqr = PDController.PD_Control_Calculate(robot_joint_ref_angle, 
                                                      robot_joint_ref_velocity,
                                                      robot_joint_feb_angl,
                                                      robot_joint_feb_vel) # calculate torque by PD controller
        
        # apply joint torque in mujoco, reshape to row vector to match the size
        mj_data.ctrl = Joint_Tqr.reshape(-1)

    
        viewer.sync()
        save_data.append([np.concatenate(([simu_time], robot_joint_ref_angle.flatten(), robot_joint_feb_angl.flatten(), Joint_Tqr.flatten()))])
    viewer.close()
    data_array = np.vstack(save_data)
    np.save('test_sin_data.npy', data_array)
