"""
创建一个评估脚本，目的是评估模型的性能
"""
from RL_Env import PandaRLEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv,VecMonitor
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
plt.rcParams['font.family']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def eval01():
    model=SAC.load('Save_Model/sac_panda_10560000_steps.zip', model_class=SAC, device='cpu')
    RENDER_MODE='human'
    env=DummyVecEnv([lambda:PandaRLEnv(render_mode=RENDER_MODE,halfs_x=0.1,halfs_y=0.2)])
    env=VecNormalize.load('Save_Model/sac_panda_vecnormalize_10560000_steps.pkl', env)
    env.training=False
    env.norm_reward=False

    num_eposide=20
    success_count=0
    total_rewards=[]
    episode_lengths=[]

    for episode in range(num_eposide):
        obs = env.reset()
        done=False
        eposide_reward=0.0
        step_count=0
        frame=[]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done,infos = env.step(action)
            info=infos[0]
            eposide_reward+=reward
            step_count+=1

            if RENDER_MODE=='rgb_array':
                frame.append(env.render())
            elif RENDER_MODE=='human':
                env.render()
                # time.sleep(0.02 )

            if info['is_success']:
                success_count+=1

        total_rewards.append(eposide_reward)
        episode_lengths.append(step_count)
        print(f"Episode {episode+1:2d} | Reward: {eposide_reward.item():8.2f} | Steps: {step_count:4d} | Grasped: {info['graspped']} |distanceToplace: {info['distanceToplace']:.4f} | Success: {info['is_success']}")

        if RENDER_MODE=='rgb_array' and len(frame)>0:
            imageio.mimsave(f'./episode_{episode+1:02d}.gif',frame,fps=50)

    print("\n================== 评估结果 ==================")
    print(f"总回合数: {num_eposide}")
    print(f"成功率: {success_count / num_eposide * 100:.1f}%")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=============================================")

    #绘图
    eposides=np.arange(1,num_eposide+1) #生成1到20的整数numpy数组
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    axes[0].plot(eposides,total_rewards)
    axes[0].set_xlabel('回合数')
    axes[0].set_ylabel('奖励')
    axes[0].set_title('回合-奖励')
    axes[1].plot(eposides,episode_lengths)
    axes[1].set_xlabel('回合数')
    axes[1].set_ylabel('步数')
    axes[1].set_title('回合-步数')
    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == '__main__':
    eval01()
