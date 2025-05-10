# H2O_sim
sim2sim.py为运行的mujoco脚本，前提是安装了mujoco

sim2sim_class.py被上述文件调用，里面需要先安装phc smpl等库，参考h2o那篇安装

g1_scene和g1_w_imu_3_balls是模型文件，用于mujoco的模型导入

ACCAD_g1_Male2MartialArtsPunches_c3d_E7uppercutleft_poses.pkl为重定向好的模型文件，来自AMASS库

25_04_06_18-19-05_g1_add_ref_vel_and_ref_delta_posmodel_20000.pt为训练好的模型

# FAQ
由于原作者的task_obs是考虑全局信息的，这里我将原来的task_obs函数观测作了以下更改：
```
# 只返回参考动作相对于参考动作root坐标系下的坐标：
def compute_imitation_observations_teleop_no_RootPos(root_rot ,ref_root_pos, ref_body_pos, ref_body_vel, body_pos, time_steps):
    obs = []
    B, J, _ = body_pos.shape  #B为环境数 ，J为track_link_ids的数目 + head  or hands

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    # heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    # heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    ##### 参考位置相对于参考root坐标系下的值
    delta_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - ref_root_pos.view(B, 1, 1, 3)  # preserves the body position
    # make some changes to how futures are appended.
    obs.append(delta_ref_body_pos.view(B, time_steps, -1))  # 1 * timestep * J * 3
    obs.append(ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3 
    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs 
```
对应于sim2sim.py文件的：
```
    def compute_imitation_observations_teleop_max_no_RootPos(self, ref_root_pos, ref_body_pos, ref_body_vel, root_rot, time_steps):
        obs = []
        B = 1
        J = len(self._track_bodies_extend_id)
        # B, J, _ = body_pos.shape  #B为环境数 ，J为track_link_ids的数目 + head  or hands   
        # ================
        heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)   #0
        # heading_inv_ref_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0) # 1 3 4 
        # heading_inv_ref_rot_expand = heading_inv_ref_rot.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0) # 1 3 4 
        # ================
        # T_quat = self.quaternion_transform(heading_inv_rot, heading_inv_ref_rot)    # 1 3 4
        # T_quat = T_quat.unsqueeze(-2).repeat((1, J, 1)).repeat_interleave(time_steps, 0)
        # ================ 
        local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - ref_root_pos.view(B, 1, 1, 3)  # preserves the body position
        local_ref_body_pos_obs = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
        local_ref_body_vel_obs = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))
        # ================
        obs.append(local_ref_body_pos.view(B, time_steps, -1))  # 1 * timestep * J * 3
        obs.append(ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3
        obs = torch.cat(obs, dim=-1).view(B, -1)
        return obs 
```
## 里面有一些路径需要自行修改：

sim2sim_class.py里的：
```
    def _load_motion(self):
        # motion文件安
        motion_path   = f'{MUJOCO_TREE_ROOT_DIR}/motions/ACCAD_g1_Male2MartialArtsPunches_c3d_E7uppercutleft_poses.pkl'  
        # urdf文件安   
        skeleton_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_phc.xml'
```
同样sim2sim_class.py里：
```
class Sim2simCfg():
    class sim_config:
        mujoco_model_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_scene.xml'
```
在sim2sim.py文件下：
```
if __name__ == '__main__':

    load_model = f'{MUJOCO_TREE_ROOT_DIR}/models/STUDENT/25_04_06_18-19-05_g1_add_ref_vel_and_ref_delta_posmodel_20000.pt'
    policy = torch.jit.load(load_model)  
    cfg = Sim2simCfg()
    cfg.sim_config.mujoco_model_path = f'{MUJOCO_TREE_ROOT_DIR}/unitree_robots/g1_19dof/xml/g1_scene.xml'
    run_mujoco(policy, cfg)
```

## ～～由于个人主用Gitlab，暂时没有维护Github，这里暂时摘要了比较重要的文件给大家参考，等抽空写完SDK部署是先后会一一整理～～
# 为避免覆盖，笔者另建新的维护项目：https://github.com/Ethereal1679/H2O_Sim_Real_reshuffle，此项目不再维护
