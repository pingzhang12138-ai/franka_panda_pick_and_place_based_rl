from RL_Env import PandaRLEnv
env=PandaRLEnv(render_mode='human')
obs,info=env.reset()
print(f'obs.shape:{obs.shape}')
print(f'distance:{obs[-1]}')

for epoch in range(100):
    env.reset()
    for step in range(100):
        action=env.action_space.sample()
        obs,reward,terminated,truncated,info=env.step(action)
        # print(f'step:{step}, reward:{reward}, terminated:{terminated}, truncated:{truncated},grasped={info["graspped"]},distanceToplace={info["distanceToplace"]:.4f}')
        env.render()
        if terminated or truncated:
            print('Episode finished')
            obs,_ =env.reset()
