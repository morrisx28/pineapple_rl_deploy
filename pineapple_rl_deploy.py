import time
import sys
import numpy as np
import threading
import traceback
import yaml
import csv
import argparse
import matplotlib.pyplot as plt # Import for plotting
import torch
# import gui_teleop_v2_1 as gui_teleop
from headless_teleop import HeadlessTeleop

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_MOTORS = 8

def apply_diamond_constraint(cmd, max_lin, max_ang):
    """
    Applies a diamond-shaped constraint (L1 norm) on linear and angular commands:
    |v_x| / v_max + |w_z| / w_max <= 1
    """
    vx = cmd[0]
    wz = cmd[2]

    limit_vx = max_lin
    limit_wz = max_ang

    if limit_vx < 1e-6:
        limit_vx = 1.0
    if limit_wz < 1e-6:
        limit_wz = 1.0

    ratio = abs(vx) / limit_vx + abs(wz) / limit_wz
    if ratio > 1.0:
        scaling = 1.0 / ratio
        cmd[0] *= scaling
        cmd[2] *= scaling

    return cmd

class Filter:
    def __init__(self, alpha):
        self.filter_value = None
        self.alpha = alpha
    
    def filt(self, input):
        if self.filter_value is None:
            self.filter_value = input
        else:
            self.filter_value = self.alpha * input + (1 - self.alpha) * self.filter_value
        return self.filter_value

class Controller:
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("config_file", type=str, help="config file name in the config folder")
        args = parser.parse_args()
        config_file = args.config_file
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            policy_path = config["policy_path"]
            self.policy = torch.jit.load(policy_path)
            self.dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)

            self.lin_vel_scale = config["lin_vel_scale"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.pos_action_scale = config["pos_action_scale"]
            self.vel_action_scale = config["vel_action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            # Optional low-pass filtering for IMU angular velocity.
            self.use_ang_vel_filter = config.get("use_ang_vel_filter", False)
            self.ang_vel_filter_alpha = float(config.get("ang_vel_filter_alpha", 0.95))

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            one_step_obs_size = config["one_step_obs_size"]
            obs_buffer_size = config.get("obs_buffer_size", 1)

            self.leg_joint_indices = config["leg_joint_indices"]
            self.wheel_joint_indices = config["wheel_joint_indices"]
            cmd = np.array(config["cmd_init"], dtype=np.float32)

            max_lin = config.get("max_lin_vel", 1.0)
            max_ang = config.get("max_ang_vel", 1.0)
            self.max_lin = max_lin
            self.max_ang = max_ang

            # Optional base-height command (enabled for height-conditioned policies).
            self.use_height_command = config.get("use_height_command", False)
            self.height_scale = config.get("height_scale", 1.0)
            self.cmd_height_init = config.get("cmd_height_init", 0.3)
            self.min_height = config.get("min_height", 0.2)
            self.max_height = config.get("max_height", 0.35)
            self.height_step = config.get("height_step", 0.005)

            self.policy_index_map = config.get("policy_index_map", None)
            if self.policy_index_map is not None:
                self.policy_index_map = np.array(self.policy_index_map, dtype=np.int64)

        if self.use_height_command:
            # pass
            # self.teleop = gui_teleop.GUITeleop(
            #     config_init=config["cmd_init"],
            #     max_lin=max_lin,
            #     max_ang=max_ang,
            #     height_init=self.cmd_height_init,
            #     height_step=self.height_step,
            #     min_height=self.min_height,
            #     max_height=self.max_height,
            # )
            self.teleop = HeadlessTeleop(
                config_init=config["cmd_init"], 
                max_lin=max_lin, 
                max_ang=max_ang,
                height_init=self.cmd_height_init,
                height_step=self.height_step,
                min_height=self.min_height,
                max_height=self.max_height
            )
            print("Headless teleop initialized (no GUI window).")
        else:
            # pass
            # self.teleop = gui_teleop.GUITeleop(config_init=config["cmd_init"], max_lin=max_lin, max_ang=max_ang)
            self.teleop = HeadlessTeleop(config_init=config["cmd_init"], max_lin=max_lin, max_ang=max_ang)
        self.target_dof_pos = self.default_angles.copy()
        self.target_dof_vel = np.zeros(num_actions)
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.obs_history_buffer = torch.zeros((obs_buffer_size, one_step_obs_size))
            
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.controller_rt = 0.0
        self.is_running = False
        self.counter = 0

        self.ang_vel_data = []
        self.qtau_data = []
        self.qtau_cmd = []

        # Data logging lists for plotting (for motor 0 as an example)
        self.time_data = []
        self.qpos_data = []
        self.qvel_data = []
        self.qtau_data = []
        self.dq_cmd_data = []
        self.tau_cmd_data = []


        # thread handling
        self.lowCmdWriteThreadPtr = None

        # state
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.quat = np.zeros(4) # q_w q_x q_y q_z
        self.ang_vel_raw = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.ang_vel_filters = [Filter(self.ang_vel_filter_alpha) for _ in range(3)]

        self.mode = ''
        # self.dt = 0.001
        self.start_time = time.perf_counter() # To calculate elapsed time


        self.crc = CRC()

        self.first_logged = False
        self.second_logged = False

        # Runtime frequency logging for move mode.
        self.move_freq_log_interval = 1.0  # seconds
        self.move_loop_count = 0
        self.move_policy_update_count = 0
        self.move_freq_window_start = time.perf_counter()

    # Control methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # Init default pos #
        self.Start()
        self.start_time = time.perf_counter() # Reset start time after threads are initialized

        print("Initial Sucess !!!")

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    

    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite, daemon=True)
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        self.teleop.close()
        if self.lowCmdWriteThreadPtr:
            self.lowCmdWriteThreadPtr.join()


    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.get_current_state()
        # self.record_data() # Record data when a new state message is received
        # print(f'FL qpos {self.low_state.motor_state[0].q} FR qpos {self.low_state.motor_state[1].q} RL qpos {self.low_state.motor_state[2].q} RR qpos {self.low_state.motor_state[3].q}')
        # quat = self.low_state.imu_state.quaternion
        # ang_vel = self.low_state.imu_state.gyroscope
        # print(f'quat w: {self.quat[0]} x: {self.quat[1]} y: {self.quat[2]} z: {self.quat[3]}')
        # print(f'ang_vel x: {self.ang_vel[0]} y: {self.ang_vel[1]} z: {self.ang_vel[2]}')
    
    
    def stand(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in [0, 1, 2, 4, 5, 6]:
                self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 40
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def reset_timer(self):
        self.controller_rt = 0.0
        self.counter = 0
        self.move_loop_count = 0
        self.move_policy_update_count = 0
        self.move_freq_window_start = time.perf_counter()

    def _log_move_frequency(self, policy_updated=False):
        self.move_loop_count += 1
        if policy_updated:
            self.move_policy_update_count += 1

        now = time.perf_counter()
        elapsed = now - self.move_freq_window_start
        if elapsed >= self.move_freq_log_interval:
            loop_hz = self.move_loop_count / elapsed
            policy_hz = self.move_policy_update_count / elapsed
            print(f"[move freq] loop: {loop_hz:.2f} Hz | policy: {policy_hz:.2f} Hz")

            self.move_loop_count = 0
            self.move_policy_update_count = 0
            self.move_freq_window_start = now
    
    def sit(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.sit_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def move(self):
        policy_updated = False
        if self.counter % self.control_decimation == 0 and self.counter > 0:

            if self.policy_index_map is not None:
                qpos_obs = self.qpos[self.policy_index_map]
                qvel_obs = self.qvel[self.policy_index_map]
                default_angles_pol = self.default_angles[self.policy_index_map]
            else:
                qpos_obs = self.qpos
                qvel_obs = self.qvel
                default_angles_pol = self.default_angles

            current_cmd_vel = np.array(self.teleop.get_command(), dtype=np.float32)
            current_cmd_vel = apply_diamond_constraint(current_cmd_vel, self.max_lin, self.max_ang)
            # current_cmd_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if self.use_height_command:
                current_cmd_height = np.array([self.teleop.get_height_command()], dtype=np.float32)
                # current_cmd_height = np.array([0.3], dtype=np.float32)

            gravity_b = self.get_gravity_orientation(self.quat)

            # SAFE leg joint delta (1D)
            valid_leg_idx = [i for i in self.leg_joint_indices if i < len(qpos_obs) and i < len(default_angles_pol)]
            leg_pos_delta = (qpos_obs[valid_leg_idx] - default_angles_pol[valid_leg_idx]) * self.dof_pos_scale
            leg_pos_delta = leg_pos_delta.astype(np.float32).ravel()
            
            obs_list = [
                self.ang_vel.copy() * self.ang_vel_scale,
                gravity_b,
                current_cmd_vel * self.cmd_scale,
            ]
            if self.use_height_command:
                obs_list.append(current_cmd_height * self.height_scale)
            obs_list.extend([
                leg_pos_delta,
                qvel_obs * self.dof_vel_scale,
                self.action.astype(np.float32).copy(),
            ])

            obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]

            current_obs = torch.cat(obs_list, dim=0)

            self.obs_history_buffer = torch.roll(self.obs_history_buffer, shifts=-1, dims=0)
            self.obs_history_buffer[-1] = current_obs

            # Stack by feature group: [feat1_t, feat1_t-1, ..., feat2_t, ...]
            split_sizes = [o.numel() for o in obs_list]
            feature_groups = torch.split(self.obs_history_buffer, split_sizes, dim=1)
            flat_groups = [g.flatten() for g in feature_groups]
            
            obs_tensor_buf = torch.cat(flat_groups).unsqueeze(0)
            obs_tensor_buf = torch.clip(obs_tensor_buf, -100, 100)

            # obs inference
            self.action = self.policy(obs_tensor_buf).detach().numpy().squeeze()
            policy_updated = True

            # Set leg joint target positions
            for idx in self.leg_joint_indices:       # 0 1 2 3 4 5
                if self.policy_index_map is not None:
                    idx_xml = self.policy_index_map[idx]
                else:
                    idx_xml = idx

                if idx_xml < len(self.target_dof_pos) and idx < len(self.action):
                    self.target_dof_pos[idx_xml] = self.default_angles[idx_xml] + self.action[idx] * self.pos_action_scale
            # Set wheel joint target velocities
            for idx in self.wheel_joint_indices:
                if self.policy_index_map is not None:
                    idx_xml = self.policy_index_map[idx]
                else:
                    idx_xml = idx

                if idx_xml < len(self.target_dof_vel) and idx < len(self.action):
                    self.target_dof_vel[idx_xml] = self.action[idx] * self.vel_action_scale

            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = self.target_dof_vel[i]
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0

            # if not self.first_logged:
            #     print("First action command sent: ", time.time())
            #     self.first_logged = True
            # if self.first_logged and not self.second_logged and self.qvel[3]>1.0:
            #     print("Second action command sent (robot starts moving): ", time.time())
            #     self.second_logged = True
        # self._log_move_frequency(policy_updated=policy_updated)
        self.counter += 1
    

    def stand_up(self):
        self.mode = 'stand'
        self.reset_timer()

    def sit_down(self):
        self.mode = 'sit'
        self.reset_timer()
    
    def move_rl(self):
        self.mode = 'move'
        self.reset_timer()


    def get_current_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
            self.qtau[i] = self.low_state.motor_state[i].tau_est

        for i in range(3):
            gyro_i = self.low_state.imu_state.gyroscope[i]
            self.ang_vel_raw[i] = gyro_i
            if self.use_ang_vel_filter:
                self.ang_vel[i] = self.ang_vel_filters[i].filt(gyro_i)
            else:
                self.ang_vel[i] = gyro_i
        
        # print("angular vel: ", self.ang_vel)

        for i in range(4):
            self.quat[i] = self.low_state.imu_state.quaternion[i]
    

    def record_data(self):
        """Records current state and command data for plotting."""
        # Use motor 0 for plotting as an example
        motor_idx = 3
        
        current_time = time.perf_counter() - self.start_time
        
        self.time_data.append(current_time)
        self.qpos_data.append(self.qpos[motor_idx])
        self.qvel_data.append(self.qvel[motor_idx])
        self.qtau_data.append(self.qtau[motor_idx])
        
        # Record command for the active motor
        self.dq_cmd_data.append(self.low_cmd.motor_cmd[motor_idx].dq)
        self.tau_cmd_data.append(self.low_cmd.motor_cmd[motor_idx].tau)



    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'stand':
                self.stand()
            elif self.mode == 'sit':
                self.sit()
            elif self.mode == 'move':
                self.move()
            
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    # if len(sys.argv)>1:
    #     ChannelFactoryInitialize(1, sys.argv[1])
    # else:
    #     ChannelFactoryInitialize(1, "lo") # default DDS port for pineapple
    ChannelFactoryInitialize(1, "eth0")
    # ChannelFactoryInitialize(1, "lo")
    controller = Controller()
    controller.Init()

    command_dict = {
        "stand": controller.stand_up,
        "sit": controller.sit_down,
        "move": controller.move_rl,
    }

    while True:        
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break

        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(-1)
