import gym
from gym import error, spaces, utils
from gym.utils import seeding

# import uav_config as uavcfg
import numpy as np
import os
import random
import datetime


# 无人机场景的参数
class UavConfig:
    def __init__(self, train_flag):
        # 路径
        self.train_flag = train_flag
        self.train_time = 40000
        # 场景的长度
        # 场景的长度
        self.total_x = 1000
        self.total_y = 1000
        # 用户随机的高度
        self.IOTD_Z = 250
        # UAV最高高度和最低高度
        self.h_min = 250
        self.h_max = 750
        # 用户数量和初始位置,随机生成用户位置
        self.iotd_num = 15
        self.fire_num = 0
        # 每个用户有最低的
        # 无人机数量和初始位置
        self.UAV_NUM = 1
        self.UAV_INITIAL_POSITION_X = 0
        self.UAV_INITIAL_POSITION_Y = 0
        self.UAV_INITIAL_POSITION_Z = self.h_min

        self.UAV_INFO_VEC = 6  # [x, y, z, uav_v, uav_dir, dir_z]
        self.STATE_DIM = self.UAV_INFO_VEC + 8 * self.iotd_num + 4 * self.fire_num
        self.ACTION_DIM = 4  # [uav_v, uav_dir, dir_z] [uav_v, uav_dir, dir_z, power] [uav_v, uav_dir, dir_z, iotd_num, power]
        self.ACTION_BOUND = [-1, 1]
        # 时隙
        self.t_max = 160
        # 无人机距离的最大值和速度最大值
        self.V_MAX = 50
        self.V_A_MAX = 20
        self.R_min = 36

        # 无人机数据 -------------------------------------------------------------------
        # 选择的功率最大值
        self.P_MAX = 26  # dBm
        self.P_min = -70  # dBm
        # NLOS非视距参数和LOS视距参数
        self.area = 'urban'
        if self.area == 'suburban':
            # suburban
            self.LOS = 0.1
            self.NLOS = 21
        elif self.area == 'urban':
            # urban
            self.LOS = 1
            self.NLOS = 20
        elif self.area == 'Dense urban':
            # Dense urban
            self.LOS = 1.6
            self.NLOS = 23
        elif self.area == 'Highrise urban':
            # Highrise urban
            self.LOS = 2.3
            self.NLOS = 34
        # A 计算
        self.A = self.LOS - self.NLOS
        #
        self.c = 3 * (10 ** 8)
        # noise
        self.a = 9.61
        self.b = 0.16
        self.BindWidth = 1
        self.fc = 2000 * (10 ** 6)
        self.B = 20 * np.log10(4 * np.pi * self.fc / self.c) + self.NLOS
        self.N0 = 10 ** (-104 / 10) * 1e-3
        # 保存系数 ------------------------------------------------------------------------
        self.save_step = 5000
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        year = str(year)
        if month < 10:
            month = '0' + str(month)
        else:
            month = str(month)
        if day < 10:
            day = '0' + str(day)
        else:
            day = str(day)
        date = year + month + day + '/'
        date = '20220411' + '/00/'
        if self.train_flag == 1:
            self.save_step = 5000
            self.save_model_com_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/com'
            self.save_model_power_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/power'
            self.save_model_track_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/track'
            self.save_model_reward_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/reward'
            self.save_model_com_num_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/com_num'
            self.save_model_r_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/r'
            self.save_model_payload_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/payload'
            self.save_model_endt_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/endt'
            self.save_model_uavv_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uavv'
            self.save_model_uava_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uava'
            self.save_model_uav_theta_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uav_theta'
            self.save_model_uav_elevation_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/uav_elevation'
            self.save_model_reward1_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward1'
            self.save_model_reward2_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward2'
            self.save_model_reward3_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward3'
            self.save_model_reward4_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward4'
            self.save_model_reward5_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward5'
            self.save_model_reward6_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward6'
            self.save_model_reward7_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'train/every_reward/reward7'
        elif train_flag == 0:
            self.save_step = 100
            self.save_model_com_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/com'
            self.save_model_power_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/power'
            self.save_model_track_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/track'
            self.save_model_reward_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/reward'
            self.save_model_com_num_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/com_num'
            self.save_model_r_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/r'
            self.save_model_payload_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/payload'
            self.save_model_endt_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/endt'
            self.save_model_uavv_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/uavv'
            self.save_model_uava_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/uava'
            self.save_model_uav_theta_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/uav_theta'
            self.save_model_uav_elevation_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/uav_elevation'
            self.save_model_reward1_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward1'
            self.save_model_reward2_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward2'
            self.save_model_reward3_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward3'
            self.save_model_reward4_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward4'
            self.save_model_reward5_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward5'
            self.save_model_reward6_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward6'
            self.save_model_reward7_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (
                self.iotd_num, self.total_x) + 'evaluate/every_reward/reward7'
        if train_flag == 0 or train_flag == 1:
            if not os.path.exists(self.save_model_com_path):
                os.makedirs(self.save_model_com_path)
            if not os.path.exists(self.save_model_power_path):
                os.makedirs(self.save_model_power_path)
            if not os.path.exists(self.save_model_track_path):
                os.makedirs(self.save_model_track_path)
            if not os.path.exists(self.save_model_reward_path):
                os.makedirs(self.save_model_reward_path)
            if not os.path.exists(self.save_model_com_num_path):
                os.makedirs(self.save_model_com_num_path)
            if not os.path.exists(self.save_model_r_path):
                os.makedirs(self.save_model_r_path)
            if not os.path.exists(self.save_model_payload_path):
                os.makedirs(self.save_model_payload_path)
            if not os.path.exists(self.save_model_endt_path):
                os.makedirs(self.save_model_endt_path)
            if not os.path.exists(self.save_model_uavv_path):
                os.makedirs(self.save_model_uavv_path)
            if not os.path.exists(self.save_model_uava_path):
                os.makedirs(self.save_model_uava_path)
            if not os.path.exists(self.save_model_uav_theta_path):
                os.makedirs(self.save_model_uav_theta_path)
            if not os.path.exists(self.save_model_uav_elevation_path):
                os.makedirs(self.save_model_uav_elevation_path)
            if not os.path.exists(self.save_model_reward1_path):
                os.makedirs(self.save_model_reward1_path)
            if not os.path.exists(self.save_model_reward2_path):
                os.makedirs(self.save_model_reward2_path)
            if not os.path.exists(self.save_model_reward3_path):
                os.makedirs(self.save_model_reward3_path)
            if not os.path.exists(self.save_model_reward4_path):
                os.makedirs(self.save_model_reward4_path)
            if not os.path.exists(self.save_model_reward5_path):
                os.makedirs(self.save_model_reward5_path)
            if not os.path.exists(self.save_model_reward6_path):
                os.makedirs(self.save_model_reward6_path)
            if not os.path.exists(self.save_model_reward7_path):
                os.makedirs(self.save_model_reward7_path)

        # 归一化参数 -------------------------------------------------------------------------------------------
        self.xy_normalize_param = self.total_x  # 横坐标的最大值
        self.z_normalize_param = self.h_max  # 纵坐标的最大值
        self.v_normalize_param = self.V_MAX  # 水平速度最大值
        self.theta_normalize_param = 2 * np.pi  # 角度最大值
        self.dis_normalize_param = np.sqrt(
            np.square(self.total_x) + np.square(self.total_y) + np.square(self.h_max))  # 最大距离
        self.uav_iotd_com_flag_param = 1
        self.uav_iotd_com_num_param = self.iotd_num
        self.t_param = self.t_max
        self.uav_elevation_param = np.pi / 2

class uavEnv(gym.Env):
    def __init__(self):
        # 参数
        self.train_flag = 1
        # 顶层参数
        self.uavConfig = UavConfig(train_flag=self.train_flag)
        self.action_dim = self.uavConfig.ACTION_DIM
        self.STATE_DIM = self.uavConfig.STATE_DIM
        # 定义action_space 和 obs space
        low = 0.
        high = 1.
        s_low = np.array([low] * self.STATE_DIM)
        s_high = np.array([high] * self.STATE_DIM)
        # a_low = np.array( [low] * self.action_dim)
        # a_high = np.array( [high] * self.action_dim)
        a_low = np.array([-1, 0, -1, 0])
        a_high = np.array([1, 1, 1, 1])
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float32)
        # 场景高度设置
        self.h_max = self.uavConfig.h_max
        self.h_min = self.uavConfig.h_min
        self.total_x = self.uavConfig.total_x
        self.total_y = self.uavConfig.total_y
        self.iotd_z = self.uavConfig.IOTD_Z
        # uav的数量和初始位置
        self.uav_x_pos_init = self.uavConfig.UAV_INITIAL_POSITION_X
        self.uav_y_pos_init = self.uavConfig.UAV_INITIAL_POSITION_Y
        self.uav_z_pos_init = self.uavConfig.UAV_INITIAL_POSITION_Z
        # uav的实时位置
        self.uav_x_pos = self.uav_x_pos_init
        self.uav_y_pos = self.uav_y_pos_init
        self.uav_z_pos = self.uav_z_pos_init
        # self.ep
        self.ep = 0
        # 用户数量和位置
        self.iotd_num = self.uavConfig.iotd_num
        self.fire_num = self.uavConfig.fire_num
        self._reset_iotd_pos()
        self._reset_fire_pos()

        # 时隙总时长
        self.t_max = self.uavConfig.t_max
        # 无人机的最大移动距离和最大功率
        self.p_max = self.uavConfig.P_MAX
        self.p_min = self.uavConfig.P_min
        # 无人机运动参数
        self.v_max = self.uavConfig.V_MAX
        self.v_a_max = self.uavConfig.V_A_MAX
        self.dt = 1

        # 关于向量维度的设置
        # action dim 和 state dim 和 action bound
        self.UAV_INFO_VEC = self.uavConfig.UAV_INFO_VEC  # [x, y, z, uav_v, uav_dir, uav_v_z, dir_z]
        self.ACTION_DIM = self.uavConfig.ACTION_DIM  # [uav_v, uav_dir, dir_z, power]
        self.ACTION_BOUND = self.uavConfig.ACTION_BOUND
        # 计算r的参数
        self.A = self.uavConfig.A
        self.N0 = self.uavConfig.N0
        self.a = self.uavConfig.a
        self.b = self.uavConfig.b
        self.B = self.uavConfig.B
        self.BindWidth = self.uavConfig.BindWidth
        # 定义保存无人机信息的矢量
        self.uav_state_normalization = np.zeros(self.STATE_DIM)
        self.uav_action = np.zeros(self.ACTION_DIM)  # 固定选择最近的用户
        # 无人机通信状态
        self.uav_iotd_check_com_bool = np.zeros(self.iotd_num, dtype=np.bool)
        self.uav_iotd_com_time = np.zeros(self.iotd_num)
        self.uav_iotd_com_num = 0
        # 无人机运动状态
        self.uav_v = 0.0  # 无人机的速度和uav的
        self.uav_a = 0.0
        self.uav_vx = 0.0
        self.uav_vy = 0.0
        self.uav_vz = 0.0
        self.uav_theta = 0.0  # 无人机的角度
        self.uav_elevation = 0.0
        self.R_min = self.uavConfig.R_min
        # 定义无人机与哪个用户通信和用户通信功率
        self.choose_iotd_num = 0
        self.choose_iotd_power = 0.0
        # 定义无人机与所有用户的距离
        self.uav_iotd_horizontal_distance = np.zeros(self.iotd_num)
        self.uav_iotd_real_distance = np.zeros(self.iotd_num)
        self.uav_iotd_theta = np.zeros(self.iotd_num)
        self.uav_iotd_theta_temp = np.zeros(self.iotd_num)
        self.uav_iotd_path_loss = np.zeros(self.iotd_num)
        self.iotd_receive_power = np.zeros(self.iotd_num)
        # 归一化参数(各项标准的最大值)
        self.xy_normalization_param = self.uavConfig.xy_normalize_param  # 横坐标的最大值
        self.z_normalization_param = self.uavConfig.z_normalize_param  # 纵坐标的最大值
        self.v_normalize_param = self.uavConfig.v_normalize_param  # 水平速度最大值
        self.theta_normalize_param = self.uavConfig.theta_normalize_param  # 水平角度
        self.dis_normalize_param = self.uavConfig.dis_normalize_param  # 最大距离
        self.uav_iotd_com_flag_normalization_param = self.uavConfig.uav_iotd_com_flag_param
        self.uav_iotd_com_num_normalization_param = self.uavConfig.uav_iotd_com_num_param
        self.t_normalization_param = self.uavConfig.t_max
        self.elevation_normalize_param = self.uavConfig.uav_elevation_param
        # 路径
        self.save_step = self.uavConfig.save_step
        self.save_model_com_path = self.uavConfig.save_model_com_path
        self.save_model_power_path = self.uavConfig.save_model_power_path
        self.save_model_track_path = self.uavConfig.save_model_track_path
        self.save_model_reward_path = self.uavConfig.save_model_reward_path
        # self.save_model_orignal_reward_path = self.uavConfig.save_model_orignal_reward_path
        self.save_model_com_num_path = self.uavConfig.save_model_com_num_path
        self.save_model_R = self.uavConfig.save_model_r_path
        self.save_model_payload = self.uavConfig.save_model_payload_path
        self.save_model_endt = self.uavConfig.save_model_endt_path
        self.save_model_uavv = self.uavConfig.save_model_uavv_path
        self.save_model_uava = self.uavConfig.save_model_uava_path
        self.save_model_uav_theta = self.uavConfig.save_model_uav_theta_path
        self.save_model_uav_elevation = self.uavConfig.save_model_uav_elevation_path
        # reward1 reward2 reward3 reward4 reward5 reward6
        self.save_model_reward1_path = self.uavConfig.save_model_reward1_path
        self.save_model_reward2_path = self.uavConfig.save_model_reward2_path
        self.save_model_reward3_path = self.uavConfig.save_model_reward3_path
        self.save_model_reward4_path = self.uavConfig.save_model_reward4_path
        self.save_model_reward5_path = self.uavConfig.save_model_reward5_path
        self.save_model_reward6_path = self.uavConfig.save_model_reward6_path
        self.save_model_reward7_path = self.uavConfig.save_model_reward7_path
        # 读取R和h的关系路径
        # self._load_R_h_relation_array_function()
        # 初始化所有回合的记录
        self._reset_record_all_eposide()
        # 清理所有无人机实时变量
        self._reset_uav_realtime_state()
        # count
        self.count = 0

    def reset(self):
        self.ep = self.ep + 1
        # 时隙清零
        self._reset_t()

        # 初始化用户的位置
        # self._reset_iotd_pos()
        # self._random_add_the_user_pos()
        # 清理无人机所有实时信息
        self._reset_uav_realtime_state()
        # 初始化无人机选择用户和选择用户的功率
        self._reset_choose_user_and_power()
        # 初始化用户的连接情况和记录
        self._reset_uav_iotd_com()
        # 清零标志关系
        self._reset_uav_iotd_com_mark()
        # 清理掉距离角度
        self._reset_theta_distance_pathloss()
        # 更新距离和theta角
        self._calculate_disance_and_theta()
        # 清理R和a
        self._reset_save_uav_R_and_a()
        # 更新state和更新归一化后的state
        self._update_uav_state()
        # 清零所有回合更新变量
        self._reset_record_one_eposide()
        # 清理所有flag
        self._reset_all_flag()
        # 获得初始化的状态
        state = self.uav_state_normalization
        return np.array(state)

    def step(self, action):
        # action = np.clip(action, self.ACTION_BOUND[0], self.ACTION_BOUND[1])
        # print("self.ACTION_BOUND[0]", self.ACTION_BOUND[0])
        # print("self.ACTION_BOUND[1]", self.ACTION_BOUND[1])
        # 0、更新时隙
        self._refresh_t()
        # 1、选择水平方向速度
        self.uav_v = self._choose_uav_v(action, self.uav_v)
        # 2、选择水平方向角度
        self.uav_theta = self._choose_uav_dir(action, self.uav_theta)
        # 3、处理垂直方向的速度和方向
        self.uav_elevation = self._choose_uav_elevation(action, self.uav_elevation)
        # 4、无人机移动、坐标更新、速度更新
        self.uav_vx, self.uav_vy, self.uav_vz = self._uav_move(action, self.uav_v, self.uav_theta, self.uav_elevation)
        # 6、计算新的距离和角度
        self._calculate_disance_and_theta()
        # 7、选择用户功率
        self.choose_iotd_power = self._choose_iotd_power(action)
        # 8、选择用户并且标记用户
        # self.choose_iotd_num = self._choose_iotd(action)
        # self.choose_iotd_num = self._choose_nearest_iotd()
        # self._mark_uav_iotd_com_4()
        self._calculate_all_iotd_receve_power()
        self.choose_iotd_num_t = self._choose_all_iotd_in_range()
        self._mark_uav_iotd_com()
        # 9、计算r与用户进行通信
        self._calculate_iotd_per_R()
        # 10、计算从火源接收到的热量
        self._calculate_fire_heat()
        # 计算能量
        # self._calculate_power()
        # 10、获取是否完成
        self.uav_finish_com_task_flag, self.uav_finish_go_back_flag, self.terminated = self._check_done()
        # 11、获得新的状态
        # self._update_uav_state()            # 更新state
        self._update_uav_state()  # 更新归一化state
        # 12、根据上面的action和是否完成计算r
        self.reward = self._reward_function()
        # 13、记录轨迹和通信记录，用来后续查看
        # self._save_uav_track_and_com()
        self._save_uav_track_and_com()
        # 14、保存reward
        self._save_uav_epreward()
        # 15、返回变量
        state = self.uav_state_normalization
        reward = self.reward
        terminated = self.terminated
        return np.array(state), reward, terminated, {}

    # 初始化路径角度等信息
    def _reset_theta_distance_pathloss(self):
        self.uav_iotd_horizontal_distance = np.zeros(self.iotd_num)
        self.uav_iotd_real_distance = np.zeros(self.iotd_num)
        self.uav_iotd_theta = np.zeros(self.iotd_num)
        self.uav_iotd_theta_temp = np.zeros(self.iotd_num)
        self.uav_iotd_path_loss = np.zeros(self.iotd_num)
        self.iotd_receive_power = np.zeros(self.iotd_num)

        self.uav_iotd_horizontal_distance_s = np.zeros(self.iotd_num)
        self.uav_iotd_real_distance_s = np.zeros(self.iotd_num)
        self.uav_iotd_theta_s = np.zeros(self.iotd_num)
        self.uav_iotd_theta_temp_s = np.zeros(self.iotd_num)
        self.uav_iotd_path_loss_s = np.zeros(self.iotd_num)
        self.iotd_receive_power_s = np.zeros(self.iotd_num)

    def _reset_fire_pos(self):
        self.fire_pos = np.zeros((self.fire_num, 3))
        self.I = np.zeros(self.fire_num)
        self.fire_dis = np.zeros(self.fire_num)
        for i in range(self.fire_num):
            self.fire_pos[i][0] = np.random.randint(100, self.total_x / 2)
            self.fire_pos[i][1] = np.random.randint(100, self.total_y / 2)
            self.fire_pos[i][2] = np.random.randint(self.h_min, self.total_x / 2)
            self.fire_dis[i] = np.sqrt(
                np.square(self.fire_pos[i][0] - self.uav_x_pos) + np.square(
                    self.fire_pos[i][1] - self.uav_y_pos) + np.square(
                    self.fire_pos[i][2] - self.uav_z_pos))
        fire_pos_path = r'C:\Users\Administrator\PycharmProjects\uav_trajectory\train\pos'
        if not os.path.exists(fire_pos_path):
            os.makedirs(fire_pos_path)
        np.save(fire_pos_path + '\ fire_pos', self.fire_pos)

    def _reset_iotd_pos(self):
        self.iotd_x_pos = np.zeros(self.iotd_num)
        self.iotd_y_pos = np.zeros(self.iotd_num)
        self.iotd_z_pos = np.zeros(self.iotd_num)
        self.iotd_pos = []
        for i in range(self.iotd_num):
            self.iotd_x_pos[i] = np.random.randint(0, self.total_x)
            self.iotd_y_pos[i] = np.random.randint(0, self.total_y)
            self.iotd_z_pos[i] = np.random.randint(0, self.iotd_z)
            self.iotd_pos.append([self.iotd_x_pos[i], self.iotd_y_pos[i], self.iotd_z_pos[i]])
        iotd_pos_path = r'C:\Users\Administrator\PycharmProjects\uav_trajectory\train\pos'
        if not os.path.exists(iotd_pos_path):
            os.makedirs(iotd_pos_path)

        np.save(iotd_pos_path + '\iotd_path', np.asarray(self.iotd_pos))

    # 初始化所有实时改变的变量
    def _reset_uav_realtime_state(self):
        self.uav_v = 0.0
        self.uav_a = 0.0
        self.uav_theta = 0.0
        self.uav_elevation = 0.0
        self.uav_vx = 0.0
        self.uav_vy = 0.0
        self.uav_vz = 0.0
        self.uav_x_pos = self.uav_x_pos_init
        self.uav_y_pos = self.uav_y_pos_init
        self.uav_z_pos = self.uav_z_pos_init
        self.uav_z_dir = 0.0

    # 初始化ep
    def _reset_record_all_eposide(self):
        # 初始化一个ep
        self.ep = 0
        self.ep_reward = 0.0
        self.all_ep_reward = []
        self.all_ep_com_num = []
        self.reward1_ep = 0.0
        self.reward2_ep = 0.0
        self.reward3_ep = 0.0
        self.reward4_ep = 0.0
        self.reward5_ep = 0.0
        self.reward6_ep = 0.0
        self.all_ep_reward1 = []
        self.all_ep_reward2 = []
        self.all_ep_reward3 = []
        self.all_ep_reward4 = []
        self.all_ep_reward5 = []
        self.all_ep_reward6 = []
        self.all_ep_reward7 = []
        self.all_ep_endt = []

    # 清零回合记录参数
    def _reset_record_one_eposide(self):
        # 用来保存回合的奖励参数
        self.ep_reward = 0.
        self.reward1_ep = 0.
        self.reward2_ep = 0.
        self.reward3_ep = 0.
        self.reward4_ep = 0.
        self.reward5_ep = 0.
        self.reward6_ep = 0.
        self.reward7_ep = 0.
        self.reward_t_ep = 0.
        #  用来保存无人机
        self.save_uav_track = []
        self.save_uav_iotd_com = []
        self.save_uav_iotd_power = []
        self.save_iotd_payload = []
        self.save_iotd_payload.append(self.payload)
        self.save_uav_v = []
        self.save_uav_a = []
        self.save_uav_theta = []
        self.save_uav_elevation = []
        # punishment
        self.punishment = 0.0

    # 更新时间
    def _refresh_t(self):
        self.t += 1

    # 用来初始化t的
    def _reset_t(self):
        self.t = 0

    # 初始化uav和用户的通信标记，两个标记，一个是bool变量和1，0变量
    def _reset_uav_iotd_com(self):
        for i in range(self.iotd_num):
            self.uav_iotd_check_com_bool[i] = False
            self.uav_iotd_com_time[i] = 0
        self.uav_iotd_com_num = 0

    # 清零R和a
    def _reset_save_uav_R_and_a(self):
        self.R_total = np.zeros((self.iotd_num, self.t_max))
        # 用来剩余通信量的变量
        self.R_remain = np.zeros(self.iotd_num)
        self.finish_com_task_flag = np.zeros(self.iotd_num, dtype=np.bool)
        self.payload = np.ones(self.iotd_num) * self.R_min
        self.endt = self.t_max

    # 记录所有与用户的记录
    def _reset_uav_iotd_com_mark(self):
        # 记录与哪个用户进行通信
        self.uav_iotd_check_com_bool = np.zeros(self.iotd_num, dtype=np.bool)
        self.uav_iotd_com_time = np.zeros(self.iotd_num)

    # 清零所有标志位
    def _reset_all_flag(self):
        self.uav_finish_com_task_flag = False
        self.uav_finish_go_back_flag = False
        self.terminated = False
        self.already_give_t_reward_flag = False
        self.ahead_of_time_finish_flag = False
        self.iotd_com_last_time_finish_num = False
        self.iotd_com_finish_num = 0
        self.add_new_user_flag = False
        self.last_time_distance = -1

    # 功率清零
    def _reset_choose_user_and_power(self):
        # 初始化选择用户的
        self.choose_iotd_num = 0
        self.choose_iotd_power = 0.0

    # 记录R和a
    def _set_save_uav_R_and_a(self, R, choose_iotd_num, t):
        self.R_total[choose_iotd_num, t - 1] = R

    # 计算R
    def _calculate_sum_R_per_iotd(self):
        sum_R_per_iotd = np.zeros(self.iotd_num)
        if self.t == self.t_max:
            for i in range(self.iotd_num):
                for t_s in range(self.t_max):
                    sum_R_per_iotd[i] += self.R_total[i, t_s]
        return sum_R_per_iotd

    # 计算loss
    def _calculate_per_iotd_loss(self):
        loss = 0
        for i in range(self.iotd_num):
            if self.finish_com_task_flag[i] == True:
                loss += 0.
            else:
                loss += (self.R_remain[i] - self.R_min)
        return loss

    def _culmulate_iotd_R_2(self, r, choose_iotd_num):
        self.R_remain[choose_iotd_num] += r
        if self.R_remain[choose_iotd_num] >= self.R_min:
            self.finish_com_task_flag[choose_iotd_num] = True
            self.payload[choose_iotd_num] = 0.0
        else:
            self.finish_com_task_flag[choose_iotd_num] = False
            self.payload[choose_iotd_num] = self.R_min - self.R_remain[choose_iotd_num]

    def _calculate_iotd_per_R(self):
        if self.is_com:
            self.r = 0.0
            for i in range(len(self.choose_iotd_num)):
                p_r_temp = self.iotd_receive_power[self.choose_iotd_num[i]]
                p_r = 10 ** (p_r_temp / 10) * 1e-3
                # print("p_r={}".format(p_r))
                sinr = np.true_divide(p_r, self.N0)
                r_temp = self.BindWidth * np.log2(1 + sinr)
                # print("r_temp={}", r_temp)
                self._culmulate_iotd_R_2(r_temp, self.choose_iotd_num[i])
                self.r += r_temp
        else:
            self.r = 0.0

    def _calculate_disance_and_theta(self):
        for i in range(self.iotd_num):
            self.uav_iotd_horizontal_distance[i] = np.sqrt(np.square(self.uav_x_pos - self.iotd_x_pos[i])
                                                           + np.square(self.uav_y_pos - self.iotd_y_pos[i])) + 1
            self.uav_iotd_real_distance[i] = np.sqrt(np.square(self.uav_iotd_horizontal_distance[i])
                                                     + np.square(self.uav_z_pos - self.iotd_z_pos[i]))
            self.uav_iotd_theta[i] = np.arctan(
                np.true_divide(self.uav_z_pos - self.iotd_z_pos[i], self.uav_iotd_horizontal_distance[i]))
            self.uav_iotd_path_loss[i] = np.true_divide(self.A, 1 + self.a * np.exp(
                -self.b * (180 / np.pi * self.uav_iotd_theta[i] - self.a))) + 20 * np.log10(
                self.uav_iotd_horizontal_distance[i] / np.cos(self.uav_iotd_theta_temp[i])) + self.B

    def _calculate_fire_heat(self):
        for i in range(self.fire_num):
            self.fire_dis[i] = np.sqrt(np.square(self.fire_pos[i][0] - self.uav_x_pos) + np.square(
                self.fire_pos[i][1] - self.uav_y_pos) + np.square(self.fire_pos[i][2] - self.uav_z_pos))
            self.I[i] += 2500 / (np.pi * self.fire_dis[i] ** 2)

    def _update_uav_state(self):
        self.uav_state_normalization[0] = self.uav_x_pos / self.total_x
        self.uav_state_normalization[1] = self.uav_y_pos / self.total_y
        self.uav_state_normalization[2] = self.uav_z_pos / self.h_max
        self.uav_state_normalization[3] = self.uav_v / self.v_normalize_param
        self.uav_state_normalization[4] = self.uav_theta / self.theta_normalize_param
        self.uav_state_normalization[5] = self.uav_elevation / self.elevation_normalize_param
        for i in range(self.iotd_num):
            self.VEC = 6
            self.uav_state_normalization[self.VEC + 8 * i] = self.iotd_x_pos[i] / self.total_x
            self.uav_state_normalization[self.VEC + 8 * i + 1] = self.iotd_y_pos[i] / self.total_y
            self.uav_state_normalization[self.VEC + 8 * i + 2] = self.iotd_z_pos[i] / self.iotd_z
            self.uav_state_normalization[self.VEC + 8 * i + 3] = self.uav_iotd_check_com_bool[i] / 1.0
            self.uav_state_normalization[self.VEC + 8 * i + 4] = self.finish_com_task_flag[i] / 1.0
            self.uav_state_normalization[self.VEC + 8 * i + 5] = self.uav_iotd_theta[i] / self.elevation_normalize_param
            self.uav_state_normalization[self.VEC + 8 * i + 6] = self.uav_iotd_real_distance[
                                                                     i] / self.dis_normalize_param
            self.uav_state_normalization[self.VEC + 8 * i + 7] = self.payload[i] / self.R_min
        for j in range(self.fire_num):
            self.uav_state_normalization[self.VEC + 8 * self.iotd_num + j * 4] = self.fire_pos[j][0] / self.total_x
            self.uav_state_normalization[self.VEC + 8 * self.iotd_num + j * 4 + 1] = self.fire_pos[j][1] / self.total_y
            self.uav_state_normalization[self.VEC + 8 * self.iotd_num + j * 4 + 2] = self.fire_pos[j][2] / self.h_max
            self.uav_state_normalization[self.VEC + 8 * self.iotd_num + j * 4 + 3] = self.fire_dis[
                                                                                         j] / self.dis_normalize_param

    # 选择速度
    def _choose_uav_v(self, action, uav_v):
        # 用加速度
        if not self.uav_finish_com_task_flag:
            action_0_temp = np.clip(action[0], -1, 1)
            choose_uav_v_a = action_0_temp * self.v_a_max
            if uav_v == self.v_max and choose_uav_v_a >= 0:
                self.uav_v_a = 0
            elif uav_v == 0 and choose_uav_v_a <= 0:
                self.uav_v_a = 0
            else:
                self.uav_v_a = choose_uav_v_a
            uav_v = uav_v + self.uav_v_a
            if uav_v > self.v_max:
                uav_v = self.v_max
            if uav_v <= 0:
                uav_v = +0.0
        else:
            action_0_temp = 1.0
            choose_uav_v_a = action_0_temp * self.v_a_max
            if uav_v == self.v_max and choose_uav_v_a >= 0:
                self.uav_v_a = 0
            elif uav_v == 0 and choose_uav_v_a <= 0:
                self.uav_v_a = 0
            else:
                self.uav_v_a = choose_uav_v_a
            uav_v = uav_v + self.uav_v_a
            if uav_v > self.v_max:
                uav_v = self.v_max
            if uav_v <= 0:
                uav_v = +0.0
        return uav_v

    # 选择方向的函数
    def _choose_uav_dir(self, action, uav_theta):
        if not self.uav_finish_com_task_flag:
            action_1_temp = np.clip(action[1], 0, 1)
            choose_uav_theta = action_1_temp * 2 * np.pi
            uav_theta = uav_theta + choose_uav_theta
            uav_theta = uav_theta % (2 * np.pi)
        else:
            theta_y = self.uav_y_pos_init - self.uav_y_pos
            theta_x = self.uav_x_pos_init - self.uav_x_pos
            uav_theta = np.arctan(np.true_divide(theta_y, theta_x + 0.0001))

            if theta_x < 0 and theta_y > 0:
                uav_theta = np.pi / 2 - uav_theta
            elif theta_x < 0 and theta_y < 0:
                uav_theta = np.pi + uav_theta
            elif theta_y == 0 and theta_x < 0:
                uav_theta = np.pi
            else:
                uav_theta = uav_theta
        return uav_theta

    # 选择方向的函数
    def _choose_uav_elevation(self, action, uav_elevation):
        if not self.uav_finish_com_task_flag:
            action_2_temp = np.clip(action[2], -1, 1)
            choose_uav_elevation = action_2_temp * (np.pi / 2)
            uav_elevation = uav_elevation + choose_uav_elevation
            if uav_elevation >= 0:
                uav_elevation = uav_elevation % (np.pi / 2)
            else:
                uav_elevation = - (np.abs(uav_elevation) % (np.pi / 2))
        else:
            d = np.sqrt(np.square(self.uav_x_pos) + np.square(self.uav_y_pos))
            uav_elevation = -np.arctan(np.true_divide(self.uav_z_pos - self.uav_z_pos_init, d + 0.00001))
        return uav_elevation

    # 选择用户功率
    def _choose_iotd_power(self, action):
        action_3_temp = np.clip(action[3], 0, 1)
        choose_iotd_p = action_3_temp * self.p_max
        return choose_iotd_p

    def _uav_move(self, action, uav_v, uav_theta, uav_elevation):
        # x,y方向的速度
        v_x = uav_v * np.cos(uav_theta) * np.cos(uav_elevation)
        v_y = uav_v * np.sin(uav_theta) * np.cos(uav_elevation)
        v_z = uav_v * np.sin(uav_elevation)
        # 获得x,y方向的移动距离
        x_move = v_x * self.dt
        y_move = v_y * self.dt
        z_move = v_z * self.dt

        self.uav_x_pos += int(x_move)
        self.uav_y_pos += int(y_move)
        self.uav_z_pos += int(z_move)

        # x
        if self.uav_x_pos < 0:
            self.uav_x_pos = 0
            self.punishment = self.punishment + 1
        elif self.uav_x_pos > self.uavConfig.total_x:
            self.uav_x_pos = self.uavConfig.total_x
            self.punishment = self.punishment + 1
        # y
        if self.uav_y_pos < 0:
            self.uav_y_pos = 0
            self.punishment = self.punishment + 1
        elif self.uav_y_pos > self.uavConfig.total_y:
            self.uav_y_pos = self.uavConfig.total_y
            self.punishment = self.punishment + 1
        # z
        if self.uav_z_pos < self.h_min:
            self.uav_z_pos = self.h_min
            self.punishment = self.punishment + 1
        elif self.uav_z_pos > self.h_max:
            self.uav_z_pos = self.h_max
            self.punishment = self.punishment + 1

        return v_x, v_y, v_z

    # 选择覆盖方位得所有用户
    def _choose_all_iotd_in_range(self):
        choose_num = []
        for i in range(self.iotd_num):
            if self.iotd_receive_power[i] >= self.p_min:
                choose_num.append(i)
        # print("choose_num", choose_num)
        return choose_num

    def _calculate_all_iotd_receve_power(self):
        self.iotd_receive_power = np.array([self.choose_iotd_power] * self.iotd_num) - self.uav_iotd_path_loss

    def _mark_uav_iotd_com(self):
        self.choose_iotd_num = []
        self.is_com = False
        for i in range(len(self.choose_iotd_num_t)):
            if not self.finish_com_task_flag[self.choose_iotd_num_t[i]]:
                self.choose_iotd_num.append(self.choose_iotd_num_t[i])
                self.is_com = True

        if self.is_com:
            for i in range(len(self.choose_iotd_num)):
                self.uav_iotd_check_com_bool[self.choose_iotd_num[i]] = True
                self.uav_iotd_com_time[self.choose_iotd_num[i]] += 1

    # 检查是否完成了
    def _check_done(self):
        # 检查是否已经完成通信任务
        self.iotd_com_finish_num = np.sum(self.finish_com_task_flag == True)
        # 判断是否增加一个用户
        if self.iotd_com_finish_num == self.iotd_com_last_time_finish_num:
            self.add_new_user_flag = False
        else:
            self.add_new_user_flag = True

        self.add_new_num = self.iotd_com_finish_num - self.iotd_com_last_time_finish_num
        self.iotd_com_last_time_finish_num = self.iotd_com_finish_num

        if self.iotd_com_finish_num == self.iotd_num:
            uav_finish_com_task_flag = True
        else:
            uav_finish_com_task_flag = False

        # 检查是否已经提前完成通信任务
        if self.t < self.t_max:
            if self.uav_x_pos == self.uav_x_pos_init and self.uav_y_pos == self.uav_y_pos_init and self.uav_z_pos == self.uav_z_pos_init and uav_finish_com_task_flag == True:
                uav_finish_go_back_flag = True
                self.ahead_of_time_finish_flag = True  # 用来记录用的
                terminated = True
                self.endt = self.t
            elif self.uav_finish_com_task_flag == True and (
                    self.uav_x_pos != self.uav_x_pos_init or self.uav_y_pos != self.uav_y_pos_init or self.uav_z_pos_init != self.uav_z_pos_init):
                uav_finish_go_back_flag = False
                terminated = False
            else:
                terminated = False
                uav_finish_go_back_flag = False

        # 检查是否在时隙t_max的时候完成通信任务
        if self.t == self.t_max:
            terminated = True
            if self.uav_x_pos == self.uav_x_pos_init and self.uav_y_pos == self.uav_y_pos_init and self.uav_z_pos == self.uav_z_pos_init and uav_finish_com_task_flag == True:
                uav_finish_go_back_flag = True
            elif uav_finish_com_task_flag == True and (
                    self.uav_x_pos != self.uav_x_pos_init or self.uav_y_pos != self.uav_y_pos_init or self.uav_z_pos != self.uav_z_pos_init):
                uav_finish_go_back_flag = False
            else:
                uav_finish_go_back_flag = False

        return uav_finish_com_task_flag, uav_finish_go_back_flag, terminated

    # 获得奖励
    def _reward_function(self):
        if self.t < self.t_max:
            self.scale = 1.0
            self.scale1 = 0.000001
            reward1 = -self.scale1 * self.punishment
            scale7 = 0

            # 完成一个新的用户通信任务奖励
            if self.add_new_user_flag:
                self.scale2 = 0.000012
            else:
                self.scale2 = 0.0
            reward2 = self.scale2 * 100 * self.add_new_num * (self.t_max - self.t)

            # 提前完成所有用户的通信任务
            if self.uav_finish_com_task_flag:
                # 如果没有给过完成任务奖励，则给奖励
                if not self.already_give_t_reward_flag:
                    self.scale3 = 0.045
                    self.already_give_t_reward_flag = True
                else:
                    self.scale3 = 0.0
                if self.uav_finish_go_back_flag:
                    self.scale4 = 1
                    self.scale5 = 0.0
                else:
                    self.scale4 = 0.0
                    self.scale5 = 0.00005
            # 没有提前完成用户的通信任务
            else:
                self.scale3 = 0.0
                self.scale4 = 0.0
                self.scale5 = 0.0
            if self.uav_theta > np.pi/4 and self.uav_theta < 2*np.pi:
                self.reward6 = -0.1
            else:
                self.reward6 = 0
            for i in range(self.fire_num):
                if self.I[i] > 1:
                    scale7 = -0.0002
                    break
                else:
                    scale7 = 0

            reward3 = self.scale3 * (self.t_max - self.t)
            reward4 = self.scale4 * (self.t_max - self.t)
            real_distance, hor_distance = self._calculate_uav_initpos_distance()
            reward5_t = -real_distance
            # reward5_t = np.sqrt(np.square(self.total_x) + np.square(self.total_y) + np.square(self.h_max-self.h_min)) - real_distance
            reward5 = self.scale5 * reward5_t
            reward7 = scale7 * sum(self.I)
            reward6 = self.reward6
            # reward = self.scale * (reward1 + reward2 + reward3 + reward4 + reward5 + reward6 + reward7)
            reward = reward2 + reward4 + reward5 + reward6 + reward7

            # 当时隙处于最后时隙的状态
        else:
            self.scale = 0.0

            self.scale1 = 0.0000
            reward1 = -self.scale1 * self.punishment

            # 完成一个新的用户通信任务奖励
            if self.add_new_user_flag == True:
                self.scale2 = 0.00012
            else:
                self.scale2 = 0.0
            reward2 = self.scale2 * 100 * self.add_new_num * (self.t_max - self.t)
            # 提前完成所有用户的通信任务
            if self.uav_finish_com_task_flag:
                self.scale6 = 0.0
                # 如果没有给过完成任务奖励，则给奖励
                if not self.already_give_t_reward_flag:
                    self.scale3 = 0.045
                    self.already_give_t_reward_flag = True
                else:
                    self.scale3 = 0.0
                if self.uav_finish_go_back_flag:
                    self.scale4 = 0.0006
                    self.scale5 = 0.0
                    self.scale7 = 0.0
                else:
                    self.scale4 = 0.0
                    self.scale5 = 0.00005
                    self.scale7 = 0.0
            else:
                self.scale3 = 0.0
                self.scale4 = 0.0
                self.scale5 = 0.0
                self.scale6 = 0.025
                self.scale7 = 0.0

            reward3 = self.scale3 * (self.t_max - self.t)
            reward4 = self.scale4 * 100 * (self.t_max - self.t)
            real_distance, hor_distance = self._calculate_uav_initpos_distance()
            # reward5_t = np.sqrt(np.square(self.total_x) + np.square(self.total_y) + np.square(self.h_max-self.h_min)) - real_distance
            reward5_t = -real_distance
            reward5 = self.scale5 * reward5_t
            reward6 = self.scale6 * self._calculate_per_iotd_loss()
            reward7 = 0
            reward = self.scale * (reward1 + reward2 + reward3 + reward4 + reward5 + reward6 + reward7)

        self.reward1_ep += reward1
        self.reward2_ep += reward2
        self.reward3_ep += reward3
        self.reward4_ep += reward4
        self.reward5_ep += reward5
        self.reward6_ep += reward6
        self.reward7_ep += reward7
        self.ep_reward += reward

        return reward

    # 计算与原点的距离
    def _calculate_uav_initpos_distance(self):
        uav_initpos_distance_temp = np.sqrt(np.square(self.uav_x_pos - self.uav_x_pos_init)
                                            + np.square(self.uav_y_pos - self.uav_y_pos_init))
        uav_initpos_distance = np.sqrt(np.square(uav_initpos_distance_temp)
                                       + np.square(self.uav_z_pos - self.uav_z_pos_init))
        uav_initpos_hor_distance = uav_initpos_distance_temp
        return uav_initpos_distance, uav_initpos_hor_distance

    def _save_uav_track_and_com(self):
        save_uav_track_temp = np.hstack((self.uav_x_pos, self.uav_y_pos, self.uav_z_pos))
        self.save_uav_track.append(save_uav_track_temp)
        # self.save_uav_iotd_com.append(self.choose_iotd_num)
        self.save_uav_iotd_power.append(np.array(self.choose_iotd_power))
        self.save_iotd_payload.append(np.array(self.payload))
        self.save_uav_v.append(self.uav_v)
        self.save_uav_a.append(self.uav_v_a)
        self.save_uav_theta.append(self.uav_theta)
        self.save_uav_elevation.append(self.uav_elevation)

        if self.is_com:
            self.save_uav_iotd_com.append(self.choose_iotd_num)
        else:
            self.save_uav_iotd_com.append(np.array([100]))

        # if self.t < self.t_max and self.ep % self.save_step == 0 and self.ahead_of_time_finish_flag == True:
        if self.t < self.t_max and self.ahead_of_time_finish_flag == True:
            self.count = self.count + 1
            if self.count % 5000 == 0 and self.count >= 5000:
                save_uav_track_name = os.path.join(self.save_model_track_path, 'uav_track_%d' % self.ep)
                save_uav_iotd_com_name = os.path.join(self.save_model_com_path, 'uav_iotd_com_%d' % self.ep)
                save_uav_iotd_com_power_name = os.path.join(self.save_model_power_path,
                                                            'uav_iotd_com_power%d' % self.ep)
                save_iotd_model_payload_name = os.path.join(self.save_model_payload, 'iotd_payload%d' % self.ep)
                save_model_R_name = os.path.join(self.save_model_R, 'R_%d' % self.ep)
                save_uav_v_name = os.path.join(self.save_model_uavv, 'uav_v%d' % self.ep)
                save_uav_a_name = os.path.join(self.save_model_uava, 'uav_a%d' % self.ep)
                save_uav_theta_name = os.path.join(self.save_model_uav_theta, 'uav_theta%d' % self.ep)
                save_uav_elevation_name = os.path.join(self.save_model_uav_elevation, 'uav_elevation%d' % self.ep)
                # 保存
                np.save(save_uav_track_name, self.save_uav_track)
                np.save(save_uav_iotd_com_name, self.save_uav_iotd_com)
                np.save(save_uav_iotd_com_power_name, self.save_uav_iotd_power)
                np.save(save_iotd_model_payload_name, self.save_iotd_payload)
                np.save(save_model_R_name, self.R_remain)
                np.save(save_uav_v_name, self.save_uav_v)
                np.save(save_uav_a_name, self.save_uav_a)
                np.save(save_uav_theta_name, self.save_uav_theta)
                np.save(save_uav_elevation_name, self.save_uav_elevation)

        elif self.t == self.t_max and self.ep % self.save_step == 0:
            save_uav_track_name = os.path.join(self.save_model_track_path, 'uav_track_%d' % self.ep)
            save_uav_iotd_com_name = os.path.join(self.save_model_com_path, 'uav_iotd_com_%d' % self.ep)
            save_uav_iotd_com_power_name = os.path.join(self.save_model_power_path, 'uav_iotd_com_power%d' % self.ep)
            save_iotd_model_payload_name = os.path.join(self.save_model_payload, 'iotd_payload%d' % self.ep)
            save_model_R_name = os.path.join(self.save_model_R, 'R_%d' % self.ep)
            save_uav_v_name = os.path.join(self.save_model_uavv, 'uav_v%d' % self.ep)
            save_uav_a_name = os.path.join(self.save_model_uava, 'uav_a%d' % self.ep)
            save_uav_theta_name = os.path.join(self.save_model_uav_theta, 'uav_theta%d' % self.ep)
            save_uav_elevation_name = os.path.join(self.save_model_uav_elevation, 'uav_elevation%d' % self.ep)

            np.save(save_uav_track_name, self.save_uav_track)
            np.save(save_uav_iotd_com_name, self.save_uav_iotd_com)
            np.save(save_uav_iotd_com_power_name, self.save_uav_iotd_power)
            np.save(save_model_R_name, self.R_remain)
            np.save(save_iotd_model_payload_name, self.save_iotd_payload)
            np.save(save_uav_v_name, self.save_uav_v)
            np.save(save_uav_a_name, self.save_uav_a)
            np.save(save_uav_theta_name, self.save_uav_theta)
            np.save(save_uav_elevation_name, self.save_uav_elevation)

    # 保存每次的reward
    def _save_uav_epreward(self):
        if self.t < self.t_max and self.ahead_of_time_finish_flag == True:
            # self.all_ep_orignal_reward.append(self.ep_reward)
            self.all_ep_com_num.append(self.uav_iotd_com_num)
            # reward1 reward2 reward3 reward4 reward5 reward6
            self.all_ep_reward1.append(self.reward1_ep)
            self.all_ep_reward2.append(self.reward2_ep)
            self.all_ep_reward3.append(self.reward3_ep)
            self.all_ep_reward4.append(self.reward4_ep)
            self.all_ep_reward5.append(self.reward5_ep)
            self.all_ep_reward6.append(self.reward6_ep)
            self.all_ep_reward7.append(self.reward7_ep)
            self.all_ep_reward.append(self.ep_reward)
            self.all_ep_endt.append(self.endt)

            if self.count % 500 == 0 and self.count >= 500:
                save_uav_reward_name = os.path.join(self.save_model_reward_path, 'uav_reward_%d' % self.ep)
                # save_uav_orignal_reward_name = os.path.join(self.save_model_orignal_reward_path, 'uav_orignal_reward_%d' % (self.ep))
                save_uav_iotd_com_num_name = os.path.join(self.save_model_com_num_path,
                                                          'uav_iotd_com_num_%d' % self.ep)
                save_endt_name = os.path.join(self.save_model_endt, 'endt_%d' % self.ep)
                np.save(save_uav_reward_name, self.all_ep_reward)
                # np.save(save_uav_orignal_reward_name, self.all_ep_orignal_reward)
                np.save(save_uav_iotd_com_num_name, self.all_ep_com_num)
                np.save(save_endt_name, self.all_ep_endt)

                # reward1 reward2 reward3 reward4 reward5 reward6
                save_uav_reward1_name = os.path.join(self.save_model_reward1_path, 'uav_reward1_%d' % self.ep)
                save_uav_reward2_name = os.path.join(self.save_model_reward2_path, 'uav_reward2_%d' % self.ep)
                save_uav_reward3_name = os.path.join(self.save_model_reward3_path, 'uav_reward3_%d' % self.ep)
                save_uav_reward4_name = os.path.join(self.save_model_reward4_path, 'uav_reward4_%d' % self.ep)
                save_uav_reward5_name = os.path.join(self.save_model_reward5_path, 'uav_reward5_%d' % self.ep)
                save_uav_reward6_name = os.path.join(self.save_model_reward6_path, 'uav_reward6_%d' % self.ep)
                save_uav_reward7_name = os.path.join(self.save_model_reward7_path, 'uav_reward7_%d' % self.ep)

                np.save(save_uav_reward1_name, self.all_ep_reward1)
                np.save(save_uav_reward2_name, self.all_ep_reward2)
                np.save(save_uav_reward3_name, self.all_ep_reward3)
                np.save(save_uav_reward4_name, self.all_ep_reward4)
                np.save(save_uav_reward5_name, self.all_ep_reward5)
                np.save(save_uav_reward6_name, self.all_ep_reward6)
                np.save(save_uav_reward7_name, self.all_ep_reward7)

        elif self.t == self.t_max:
            # self.all_ep_orignal_reward.append(self.ep_reward)
            self.all_ep_com_num.append(self.uav_iotd_com_num)
            # reward1 reward2 reward3 reward4 reward5 reward6
            self.all_ep_reward1.append(self.reward1_ep)
            self.all_ep_reward2.append(self.reward2_ep)
            self.all_ep_reward3.append(self.reward3_ep)
            self.all_ep_reward4.append(self.reward4_ep)
            self.all_ep_reward5.append(self.reward5_ep)
            self.all_ep_reward6.append(self.reward6_ep)
            self.all_ep_reward7.append(self.reward7_ep)
            self.all_ep_reward.append(self.ep_reward)
            self.all_ep_endt.append(self.endt)

            if self.ep % self.save_step == 0:
                save_uav_reward_name = os.path.join(self.save_model_reward_path, 'uav_reward_%d' % self.ep)
                # save_uav_orignal_reward_name = os.path.join(self.save_model_orignal_reward_path, 'uav_orignal_reward_%d' % (self.ep))
                save_uav_iotd_com_num_name = os.path.join(self.save_model_com_num_path,
                                                          'uav_iotd_com_num_%d' % self.ep)
                save_endt_name = os.path.join(self.save_model_endt, 'endt_%d' % self.ep)

                np.save(save_uav_reward_name, self.all_ep_reward)
                # np.save(save_uav_orignal_reward_name, self.all_ep_orignal_reward)
                np.save(save_uav_iotd_com_num_name, self.all_ep_com_num)
                np.save(save_endt_name, self.all_ep_endt)

                # reward1 reward2 reward3 reward4 reward5 reward6
                save_uav_reward1_name = os.path.join(self.save_model_reward1_path, 'uav_reward1_%d' % self.ep)
                save_uav_reward2_name = os.path.join(self.save_model_reward2_path, 'uav_reward2_%d' % self.ep)
                save_uav_reward3_name = os.path.join(self.save_model_reward3_path, 'uav_reward3_%d' % self.ep)
                save_uav_reward4_name = os.path.join(self.save_model_reward4_path, 'uav_reward4_%d' % self.ep)
                save_uav_reward5_name = os.path.join(self.save_model_reward5_path, 'uav_reward5_%d' % self.ep)
                save_uav_reward6_name = os.path.join(self.save_model_reward6_path, 'uav_reward6_%d' % self.ep)
                save_uav_reward7_name = os.path.join(self.save_model_reward7_path, 'uav_reward7_%d' % self.ep)

                np.save(save_uav_reward1_name, self.all_ep_reward1)
                np.save(save_uav_reward2_name, self.all_ep_reward2)
                np.save(save_uav_reward3_name, self.all_ep_reward3)
                np.save(save_uav_reward4_name, self.all_ep_reward4)
                np.save(save_uav_reward5_name, self.all_ep_reward5)
                np.save(save_uav_reward6_name, self.all_ep_reward6)
                np.save(save_uav_reward7_name, self.all_ep_reward7)
