import mujoco.viewer
import numpy as np


model=mujoco.MjModel.from_xml_path(r'D:\AAA_my_workspace\Model_Demo\Compliance_Arm_RL\panda_rl_env.xml')
data=mujoco.MjData(model)

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        print(f'x:{data.geom("target_geom").xpos[0]:.4f},y:{data.geom("target_geom").xpos[1]:.4f}')
        mujoco.mj_step(model, data)
        viewer.sync()





