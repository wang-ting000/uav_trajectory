'''
作用：主要用来训练的模型的
'''
import datetime
import os
import sys
import time

import MyLogger
import uav_gym
import uav_gym.envs.uav_env
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, LstmPolicy


# from stable_baselines import DDPG
# tensorboard --logdir=user\uav\20220319\00\data\user15_dis1000\tensorboard\PPO2_1 --host=127.0.0.1

# 这里定义了mlp神经网络，中间层是256，256然后通过
class UavMlpPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(UavMlpPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[256, 256],
                                                          vf=[256, 256])],
                                           feature_extraction="mlp")


register_policy('UavMlpPolicy', UavMlpPolicy)


class UavMlpLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, **_kwargs):
        super(UavMlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                               layer_norm=False, feature_extraction="mlp", **_kwargs)


register_policy('UavMlpLstmPolicy', UavMlpLstmPolicy)


def main():
    order = '/00/'
    # 保存时间
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
    date = year + month + day + '/' + order
    date = '20220411' + '/00/'

    env = make_vec_env('uav_env-v0', n_envs=8)
    topConfig = uav_gym.envs.uav_env.UavConfig(train_flag=1)
    tensorboard_log = 'user/uav/' + date + 'data/' + \
                      'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'tensorboard/'
    model_save_path = 'user/uav/' + date + 'data/' + \
                      'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'model/' + 'uav_ppo2'
    train_log_name = 'user/uav/' + date + 'data/' + \
                     'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) + 'train_log_file'
    if not os.path.exists(train_log_name):
        os.makedirs(train_log_name)
    train_log_name += '/log_file.txt'

    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    sys.stdout = MyLogger.Logger(train_log_name)
    start_time = time.time()
    # 创建一个PPO算法模型
    model = PPO2(UavMlpPolicy, env, n_steps=topConfig.t_max, learning_rate=1e-4, tensorboard_log=tensorboard_log,
                 gamma=0.99, verbose=1, cliprange=0.20)
    # 训练
    model.learn(total_timesteps=topConfig.t_max * topConfig.train_time)
    model.save(model_save_path)
    end_time = time.time()
    run_time = end_time - start_time
    print('train finish and the time of training progress is {}s'.format(run_time))


if __name__ == '__main__':
    main()
