"""
利用RL_env的环境进行训练，创建多个环境并行训练
使用stable_baseline3中的SAC和PPO算法
"""
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback,BaseCallback
from RL_Env import PandaRLEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import multiprocessing
import numpy as np
from collections import deque
os.environ['WANDB_API_KEY']='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'


class AdaptiveSpawnCurriculum(BaseCallback):
    def __init__(self,eval_env=None,halfs_x=(0.0,0.05,0.10,0.10,0.10),halfs_y=(0.0,0.05,0.10,0.15,0.20),window=100,promote=0.60,demote=0.25):
        super().__init__()
        self.halfs_x,self.halfs_y,self.stages=list(halfs_x),list(halfs_y),0
        self.results=deque(maxlen=window)
        self.eval_env=eval_env
        self.promote,self.demote=promote,demote

    def _on_training_start(self):
        self._sync_env()

    def _sync_env(self):
        hx, hy = self.halfs_x[self.stages], self.halfs_y[self.stages]
        self.training_env.env_method('set_curriculum', hx, hy)
        if self.eval_env is not None:
            self.eval_env.env_method('set_curriculum', hx, hy)

    def _on_step(self):
        for info in self.locals.get('infos',[]):
            if info.get('episode_terminated'):
                self.results.append(bool(info.get('is_success')))
        if len(self.results)<self.results.maxlen:
            return True
        rate=np.mean(self.results)

        if rate>=self.promote and self.stages<len(self.halfs_x)-1:
            self.stages+=1
            self._sync_env()
            self.results.clear()
        elif rate<self.demote and self.stages>0:
            self.stages-=1
            self._sync_env()
            self.results.clear()
        if self.n_calls % 500 == 0:
            self.logger.record('curriculum/halfs_x',self.halfs_x[self.stages])
            self.logger.record('curriculum/halfs_y',self.halfs_y[self.stages])
            self.logger.record('curriculum/success_rate',rate)
        return True



def train01():
    wandb.init(
        project='Panda_RL',
        name='pick_and_place',
        sync_tensorboard=True, #上传tensorboard日志到wandb
        save_code=False
    )

    train_env=make_vec_env(PandaRLEnv,n_envs=16,vec_env_cls=SubprocVecEnv)
    train_env=VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_env.training=True
    train_env.norm_reward=False

    eval_env=make_vec_env(PandaRLEnv,n_envs=1,env_kwargs={'render_mode':'rgb_array'}) #评估模型创建一个新的环境
    eval_env=VecMonitor(eval_env) #监控评估环境状态
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training=False
    eval_env.norm_reward=False
    sync_envs_normalization(train_env, eval_env)

    model=SAC(
        policy='MlpPolicy', #策略网络
        env=train_env, #环境
        learning_rate=3e-4, #学习率
        buffer_size=200000, #样本池大小
        batch_size=512, #每次更新时，从经验回放缓冲区中采样 batch_size 个样本，执行 1 次梯度下降。
        gamma=0.99,# 折扣因子
        tau=0.005, #目标网络                                                                                                                                                                                                        软更新系数
        ent_coef='auto', #自动调整最大化熵系数，越大越鼓励探索
        train_freq=1, #每隔 1个环境步执行一次主网络梯度更新。
        gradient_steps=1, #每次更新时，从经验回放缓冲区中采样 batch_size 个样本，执行 1 次梯度下降。
        policy_kwargs=dict(net_arch=[512,512]), #策略网络的网络结构
        verbose=1, #打印训练日志
        tensorboard_log='./tensorboard_logs' #tensorboard日志目录
    )

    # model=PPO(
    #     policy='MlpPolicy',
    #     env=train_env,
    #     learning_rate=4e-4,
    #     n_steps=2048, #表示每个并行环境在进行一次策略更新前，需要运行的步数。总样本=n_steps*n_envs
    #     batch_size=1024,#每次策略更新用到的样本量
    #     n_epochs=10, #每批数据重复更新的次数
    #     gamma=0.95,# 折扣因子
    #     clip_range=0.1, #限制新旧策略的比例
    #     ent_coef=0.01, #熵系数
    #     gae_lambda=0.95, #平衡偏差和方差
    #     policy_kwargs=dict(net_arch=[256,256]), #策略网络的网络结构
    #     verbose=1,#打印训练日志
    #     tensorboard_log='./tensorboard_logs', #tensorboard日志目录
    #     device='cpu'
    # )

    # 每隔一段时间评估一下模型
    eval_callback=EvalCallback(
        eval_env=eval_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/eval',
        n_eval_episodes=20,
        eval_freq=20000,
        deterministic=True,
        render=True,
    )
    #每隔一段时间保存一下模型
    checkpoint_callback=CheckpointCallback(
        save_freq=20000,
        save_path='./logs/checkpoints',
        name_prefix='sac_panda',
        save_vecnormalize=True
    )

    wandb_callback=WandbCallback(
        gradient_save_freq=0, #设为0，不保存梯度
        verbose=2
    )

    curriculum_callback=AdaptiveSpawnCurriculum(eval_env=eval_env)

    model.learn(
        total_timesteps=10000000,
        callback=[eval_callback,checkpoint_callback,wandb_callback,curriculum_callback]
    )

    model.save('./logs/sac_panda_final')
    train_env.save('./logs/vec_normalize.pkl')
    wandb.finish()
    print('Training finished')


def train_resume():
    normal_path=r'./logs/checkpoints/sac_panda_vecnormalize_7680000_steps.pkl'
    model_path=r'./logs/checkpoints/sac_panda_7680000_steps.zip'

    wandb.init(
        project='Panda_RL',
        name='pick_and_place',
        id='qy7a4q6i',
        sync_tensorboard=True,
        save_code=False,
        resume='must'
    )

    train_env=make_vec_env(PandaRLEnv,n_envs=16,vec_env_cls=SubprocVecEnv)
    train_env=VecMonitor(train_env)
    train_env = VecNormalize.load(normal_path, train_env)
    train_env.training=True
    train_env.norm_reward=False

    eval_env=make_vec_env(PandaRLEnv,n_envs=1,env_kwargs={'render_mode':'human'}) #评估模型创建一个新的环境
    eval_env=VecMonitor(eval_env) #监控评估环境状态
    eval_env = VecNormalize.load(normal_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    sync_envs_normalization(train_env, eval_env)

    model=SAC.load(model_path, env=train_env)

    # 每隔一段时间评估一下模型
    eval_callback=EvalCallback(
        eval_env=eval_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/eval',
        n_eval_episodes=20,
        eval_freq=5000,
        deterministic=True,
        render=True,
    )
    #每隔一段时间保存一下模型
    checkpoint_callback=CheckpointCallback(
        save_freq=20000,
        save_path='./logs/checkpoints',
        name_prefix='sac_panda',
        save_vecnormalize=True
    )

    wandb_callback=WandbCallback(
        gradient_save_freq=0, #设为0，不保存梯度
        verbose=2
    )

    curriculum_callback = AdaptiveSpawnCurriculum(eval_env=eval_env)

    model.learn(
        total_timesteps=5000000,
        callback=[eval_callback,checkpoint_callback,wandb_callback,curriculum_callback],
        reset_num_timesteps=False
    )

    model.save('./logs/sac_panda_final')
    train_env.save('./logs/vec_normalize.pkl')
    wandb.finish()
    print('Training finished')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # train01()
    train_resume()

