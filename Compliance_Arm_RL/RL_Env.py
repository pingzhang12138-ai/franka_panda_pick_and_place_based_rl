"""
Mujoco强化学习训练策略流程：
1、确定任务目标
2、搭建仿真环境
3、封装训练环境
4、选择强化学习算法
5、训练模型并调优
6、评估模型

任务目标：利用SAC算法训练Panda臂抓取物体并放置到指定地点
此代码文件的目的是基于panda_rl_env.xml定义一个环境类，用于训练SAC算法，实现一下方法：
__init__方法：加载模型，定义动作空间和观测空间
reset方法：初始化机器人关节和物体位置
_get_obs方法：获取当前观测
step方法：执行动作，步进仿真，计算奖励和终止条件
render方法：渲染仿真结果
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import mujoco.viewer


class PandaRLEnv(gym.Env):
    metadata={'render_modes':['human','rgb_array'],"render_fps":100}
    def __init__(self,max_steps=1000,render_mode=None,halfs_x=0.0,halfs_y=0.0):
        super().__init__()
        self.xml_path=r'D:\AAA_my_workspace\Model_Demo\Compliance_Arm_RL\panda_rl_env.xml'
        self.model=mujoco.MjModel.from_xml_path(self.xml_path)
        self.data=mujoco.MjData(self.model)
        self.render_mode=render_mode
        self.renderer=None
        self.viewer=None
        #定义动作空间；space.Box表示动作是一个连续的向量，范围在-1到1之间；shape=(7,)表示动作的维度为7
        self.action_space=spaces.Box( -1.0,1.0,shape=(7,),dtype=np.float32)
         #定义观测空间；obs_dim=7+7+3+3+3+1+1+1+1+1+3+3表示观测的维度为关节角度(7)+关节速度(7)+末端位置(3)+物体位置(3)+放置点位置（3）+夹爪开度（1）+末端距物体(1)+物体距离放置点（1）+是否抓取过物体（1）+ 是否举起物体（1）+是否释放夹爪(1)+TCPz轴姿态（9）+xy轴方向物体与放置点距离（1）
        obs_dim=7+7+3+3+3+1+1+1+1+1+9+9+1+1
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(obs_dim,),dtype=np.float32)

        #self.model.nu表示模型的执行器数量；self.model.nv表示模型的自由度数;大多数机械臂nu=nv
        #控制参数
        self.max_steps=max_steps
        self.step_count=0
        self.action_scale=0.05
        self.target_threshold=0.02
        self.prev_distanceTotarget=0.0
        self.prev_distanceToplace=0.0
        self.prev_distanceToplace_xy=0.0
        self.prev_height=0.0
        self.halfs_x=halfs_x
        self.halfs_y=halfs_y

        self.grasp_confirmed=False
        self.lifted=False
        self.released=False

        self.frames_threshold=10 #连续掉线10帧才判定抓取失败（1帧≈0.02s）
        self.lost_contact_frames=0
        self.in_grasp_frames=0
        self.released_step = 0

        self.eef_site_name='TCP'
        self.eef_site_id=self.data.site(self.eef_site_name).id
        self.target_name='target'
        self.target_id=self.data.body(self.target_name).id
        self.coll1_id = self.data.geom('target_geom').id
        self.place_name='place_site'
        self.coll2_id=self.data.geom('place_bottom').id

    def _get_obs(self):
        #获取当前关节角度/关节速度/夹爪开度
        qpos=self.data.qpos[:7].copy()
        qvel=self.data.qvel[:7].copy()
        gripper_pos=self.data.qpos[7]*2
        #获取当前末端/物体/放置点位置
        eef_pos=self.data.site(self.eef_site_name).xpos.copy()
        target_pos=self.data.body(self.target_name).xpos.copy()
        place_pos=self.data.site(self.place_name).xpos.copy()
        #计算距离
        distance_effTotarget=np.linalg.norm(eef_pos-target_pos)
        distance_targetToplace=np.linalg.norm(target_pos-place_pos)
        distance_targetToplace_xy=np.linalg.norm(target_pos[:2]-place_pos[:2])
        #TCP和target姿态
        tcp_mat=self.data.site_xmat[self.eef_site_id].copy()
        target_mat=self.data.xmat[self.target_id].copy()

        return np.concatenate([qpos, qvel, eef_pos, target_pos, place_pos, [gripper_pos],[distance_targetToplace_xy], [distance_effTotarget], [distance_targetToplace], [float(self.grasp_confirmed)], [float(self.lifted)],[float(self.released)],tcp_mat,target_mat]).astype(np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        self.data.qvel[:] = 0.0

        eef_pos=self.data.site(self.eef_site_name).xpos.copy()
        self.data.qpos[9:12]=self._random_target_pos()
        self.data.qpos[12:]=[1,0,0,0]
        mujoco.mj_forward(self.model, self.data)  # 模型向前计算一步，更新所有依赖于当前状态的量

        target_pos = self.data.body('target').xpos.copy()
        place_pos=self.data.site(self.place_name).xpos.copy()
        self.prev_distanceTotarget=np.linalg.norm(eef_pos-target_pos)
        self.prev_distanceToplace=np.linalg.norm(target_pos-place_pos)
        self.prev_height=target_pos[2]
        self.prev_distanceToplace_xy=np.linalg.norm(target_pos[:2] - place_pos[:2])

        self.grasp_confirmed=False
        self.lifted=False
        self.released=False
        self.lost_contact_frames=0
        self.released_step=0
        self.in_grasp_frames=0
        self.step_count=0

        return self._get_obs(),{}

    def step(self,action):
        action=np.clip(action,-1.0,1.0)
        joint_action=action*self.action_scale
        self.data.ctrl[:7] = self.data.qpos[:7] + joint_action

        terminated = False
        is_success=False
        reward=0.0

        #物理引擎步进10步，RL网络才决策1次，这样会将RL运行在50HZ以内，动作更加平滑
        for _ in range(10):
            mujoco.mj_step(self.model,self.data)

        self.step_count+=1
        obs=self._get_obs()
        distanceToplace_xy=obs[-24]
        distanceTotarget=obs[-23]
        distanceToplace=obs[-22]
        TCP_pos=obs[14:17]
        target_pos=obs[17:20]
        place_pos=obs[20:23]
        target_mat=obs[-9:]
        tcp_mat=obs[-18:-9]
        world_z = np.array([0, 0, 1])
        target_vel = self.data.body(self.target_id).cvel[3:]

        left_touch = self.data.sensor('touch_left').data[0]
        right_touch = self.data.sensor('touch_right').data[0]

        jacobi_pos = np.zeros((3, self.model.nv))
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site_name)
        mujoco.mj_jacSite(self.model, self.data, jacobi_pos, None, site_id)
        eef_vel = jacobi_pos[:, :7] @ self.data.qvel[:7]

        if not self.grasp_confirmed:
            self.data.ctrl[7]=255
            reward_distance1=2.0*(self.prev_distanceTotarget-distanceTotarget)
            self.prev_distanceTotarget = distanceTotarget
            reward_diatance2 = -1.0*distanceTotarget

            reward_eef_vel1 = -0.1 * np.dot(eef_vel, eef_vel) * np.exp(-10 * distanceTotarget)  # 限制TCP速度

            x_tcp=tcp_mat[0:3]
            x_target=target_mat[0:3]
            reward_orientation1=0.1*np.dot(x_tcp,x_target)*np.exp(-distanceTotarget)

            z_tcp=tcp_mat[6:9]
            z_target=target_mat[6:9]
            reward_orientation2=-0.3*np.dot(z_tcp,z_target)*np.exp(-distanceTotarget)

            reward = reward_distance1 + reward_diatance2 + reward_eef_vel1 + reward_orientation2+reward_orientation1
            
            if distanceTotarget<0.01:
                self.data.ctrl[7]=0.0
                reward += 0.05
                if left_touch > 2 and right_touch > 2:

                    reward += 5.0 * (target_pos[2] - self.prev_height)
                    self.prev_height = target_pos[2]

                    if not self.grasp_confirmed and target_pos[2]>0.90:
                        self.in_grasp_frames+=1
                        if self.in_grasp_frames >= self.frames_threshold:
                            self.grasp_confirmed = True
                            reward+=50.0
                    else:
                        self.in_grasp_frames=0

        elif not self.lifted:

            if left_touch < 3 and right_touch < 3 and (TCP_pos[2]-target_pos[2])>0.05:
                self.lost_contact_frames += 1
                if self.lost_contact_frames >= self.frames_threshold:
                    reward -= 20.0
                    terminated = True
            else:
                self.lost_contact_frames = 0

                reward_lift1=5.0*(target_pos[2]-self.prev_height)
                self.prev_height=target_pos[2]
                height_err=abs(target_pos[2]-1.1)
                reward_lift2=-6.0*max(0,height_err-0.05)

                reward_orientation3=-0.05*np.dot(world_z,tcp_mat[6:9])

                reward_eff_vel=-0.1*np.dot(eef_vel,eef_vel)
                reward+=reward_lift1+reward_lift2+reward_eff_vel+reward_orientation3

                if height_err<0.05:
                    self.lifted=True
                    reward+=50.0
                    self.prev_distanceToplace=distanceToplace
                    self.prev_distanceToplace_xy=distanceToplace_xy

        else:

            if not self.released:
                if (left_touch < 3 and right_touch < 3 and (TCP_pos[2] - target_pos[2]) > 0.05):
                    self.lost_contact_frames += 1
                    if self.lost_contact_frames >= self.frames_threshold:
                        reward -= 20.0
                        terminated = True
                else:
                    self.lost_contact_frames = 0

                reward_distance3=2.0*(self.prev_distanceToplace_xy-distanceToplace_xy)
                self.prev_distanceToplace_xy=distanceToplace_xy
                reward_distance4=-1.0*distanceToplace_xy

                reward_eef_vel2=-0.1*np.dot(eef_vel,eef_vel)*np.exp(-5*distanceToplace)#限制TCP速度

                reward_orientation4=-0.1*np.dot(world_z,tcp_mat[6:9])

                reward += reward_distance3 + reward_distance4 + reward_eef_vel2+reward_orientation4

                blend=np.clip(distanceToplace_xy/0.08,0,1)
                z_target=place_pos[2]+blend*(1.1-place_pos[2])
                reward-=1.0*abs(target_pos[2]-z_target)

                if distanceToplace_xy<0.03:

                    reward += 2.0 * (self.prev_distanceToplace - distanceToplace)
                    self.prev_distanceToplace = distanceToplace

                    if distanceToplace<0.04:
                            reward+=20.0
                            self.data.ctrl[7]=255
                            self.released=True
                            self.released_step=self.step_count

            else:
                in_box = distanceToplace_xy < 0.03 and 0.84 < target_pos[2] < 0.90 and np.linalg.norm(target_vel) < 0.05
                if  self.step_count-self.released_step>self.frames_threshold*3:
                    reward-=40.0
                    terminated=True

                elif self.step_count-self.released_step>self.frames_threshold and in_box:
                    reward += 100.0
                    terminated = True
                    is_success = True


        reward-=0.001*np.dot(action,action)
        reward-=0.001
        truncated=self.step_count>=self.max_steps
        if self.data.body(self.target_id).xpos[2]<0.5:
            terminated=True
            reward-=20.0

        info={
              "graspped":self.grasp_confirmed,
              "distanceTotarget":distanceTotarget,
              "distanceToplace":distanceToplace,
              "is_success":is_success,
              "episode_terminated":terminated or truncated
        }
        return obs,reward,terminated,truncated,info

    def render(self):
        #viewer模式是创建一个窗口，用于显示仿真结果；renderer模式是将仿真结果渲染为RGB数组，不创建窗口
        if self.render_mode=='human':
            if self.viewer is None:
                self.viewer=mujoco.viewer.launch_passive(self.model,self.data)
            self.viewer.sync()
        elif self.render_mode=='rgb_array':
            if self.renderer is None:
                self.renderer=mujoco.Renderer(self.model,height=480,width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def set_curriculum(self, halfs_x, halfs_y):
        self.halfs_x = halfs_x
        self.halfs_y = halfs_y

    def _random_target_pos(self):
        x=np.random.uniform(-self.halfs_x,self.halfs_x)+0.5
        y=np.random.uniform(-self.halfs_y,self.halfs_y)+0.3
        return np.array([x,y,0.83],dtype=np.float32)
    #检测碰撞
    def _detect_collision(self,coll1_id,coll2_id):
        for contact in self.data.contact:
            if (contact.geom1==coll1_id and contact.geom2==coll2_id) or (contact.geom1==coll2_id and contact.geom2==coll1_id):
                return True
        return False




