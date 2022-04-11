'''
作用：这个文件主要是用来测试模型的读取模型，然后运行几百次，取平均值，这个模型是用来评价基于PPO算法的模型的
'''
import os
import sys
import time

import gym
import numpy as np
from stable_baselines import PPO2

from train import MyLogger
import uav_gym
import uav_gym.envs.uav_env


# 这个文件是用来测试模型的
def evaluate(env, model, topConfig, num_steps, date):
    all_episode_rewards = []
    all_episode_time = []
    ahead_finish_time = 0
    min_time_index = 0
    save_obs = np.zeros((num_steps, 0)).tolist()
    for episode in range(1, num_steps + 1):
        dones = False
        obs = env.reset()
        for i in range(topConfig.STATE_DIM):
            save_obs[episode - 1].append(obs[i])
        episode_rewards = 0.
        t = 0.
        while dones is False:
            t += 1
            action, _states = model.predict(obs)
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, rewards, dones, info = env.step(clipped_action)
            # obs, rewards, dones, info = env.step(action)
            for i in range(topConfig.STATE_DIM):
                save_obs[episode - 1].append(obs[i])
            episode_rewards += rewards

        print('episode is: {}, time step is: {} episode_rewards is: {}'.format(episode, t, episode_rewards))
        all_episode_rewards.append(episode_rewards)
        all_episode_time.append(t)
        if t < topConfig.t_max:
            ahead_finish_time = ahead_finish_time + 1

        min_time_index = np.argmin(all_episode_time)
        if all_episode_time[min_time_index] == 160:
            min_time_index = np.argmax(all_episode_rewards)

    average_rewards = sum(all_episode_rewards) / len(all_episode_rewards)
    average_time_step = sum(all_episode_time) / len(all_episode_time)

    print('finish evaluate the model!!')
    print('the average_rewards is {},\r\n'
          'and average_time_step is {},\r\n'
          'and finish all task before T_MAX is {},\r\n,'
          'and the min time is {} and index is {}.\r\n'.
          format(average_rewards,
                 average_time_step,
                 ahead_finish_time / num_steps,
                 all_episode_time[min_time_index], min_time_index + 1))

    save_obs_path = 'user/uav/' + date + '/data/' + 'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) \
                    + 'evaluate_log_file'
    if not os.path.exists(save_obs_path):
        os.makedirs(save_obs_path)
    save_obs_path += '/evaluate_file.txt'
    np.savetxt(save_obs_path, save_obs[min_time_index], delimiter=',', fmt='%s')


def main():
    topConfig = uav_gym.envs.uav_env.UavConfig(train_flag=False)
    date1 = r'20220411\00'
    date = '20220411/00/'

    evaluate_log_name = '/user/uav/' + date + 'data/' + 'user%d_dis%d/' % (topConfig.iotd_num, topConfig.total_x) \
                        + 'evaluate_log_file'
    if not os.path.exists(evaluate_log_name):
        os.makedirs(evaluate_log_name)
    evaluate_log_name += '/log_file.txt'
    sys.stdout = MyLogger.Logger(evaluate_log_name)
    env = gym.make('uav_env-v0')
    # env = make_vec_env('uav_env-v0', n_envs=8)

    model_path = r'C:\Users\Administrator\PycharmProjects\uav_trajectory\train\user\uav\%s\data\user%d_dis%d\model\uav_ppo2.zip' %(date1,topConfig.iotd_num,topConfig.total_x)

    model = PPO2.load(model_path)
    start_time = time.time()
    evaluate(env=env, model=model, num_steps=500, date=date, topConfig=topConfig)
    end_time = time.time()
    run_time = end_time - start_time
    print('evaluate finish and the time of evaluate progress is {}s'.format(run_time))


if __name__ == '__main__':
    main()
