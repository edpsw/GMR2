import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print

from general_motion_retargeting import RobotMotionViewer
from scipy.spatial.transform import Rotation as R

import joblib


Q2_19dof_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
     
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    # [1, 0, 0], # l_ankle_pitch

    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    


    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    # [1, 0, 0],

    
    [0, 0, 1], # waist_yaw_joint

    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow
    ]])

Q2_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    
    
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_pitch

    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    


    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0],

    
    [0, 0, 1], # waist_yaw_joint

    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow
    ]])

G1_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_roll
    
    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0], # r_ankle_roll
    
    [0, 0, 1], # waist_yaw_joint
    [1, 0, 0], # waist_roll_joint
    [0, 1, 0], # waist_pitch_joint
   
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    
    [0, 1, 0], # l_elbow

    [1, 0, 0], # l_wrist_roll
    [0, 1, 0], # l_wrist_pitch
    [0, 0, 1], # l_wrist_yaw
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    
    [0, 1, 0], # r_elbow

    [1, 0, 0], # l_wrist_roll
    [0, 1, 0], # l_wrist_pitch
    [0, 0, 1], # l_wrist_yaw
    ]])

def foot_detect(positions, thres=0.002):
    fid_r, fid_l = [11], [5]
    positions = positions.numpy()
    velfactor, heightfactor = np.array([thres, thres]), np.array([0.1, 0.1]) 
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[1:,fid_l,2]
    # print('+++++++++++',feet_l_h)
    # print("+++++++++++++",feet_l_z.shape,velfactor.shape)
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(int) & (feet_l_h < heightfactor).astype(int)).astype(np.float32)
    feet_l = np.concatenate([np.array([[1., 1.]]),feet_l],axis=0)
    feet_l = np.max(feet_l, axis=1, keepdims=True)
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[1:,fid_r,2]
    # print('+++++++++++',feet_r_h)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(int) & (feet_r_h < heightfactor).astype(int)).astype(np.float32)
    feet_r = np.concatenate([np.array([[1., 1.]]),feet_r],axis=0)
    feet_r = np.max(feet_r, axis=1, keepdims=True)
    return feet_l, feet_r


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="motion_data/LAFAN1"
    )
    
    parser.add_argument(
        "--robot",
        choices=["qiao_q2_19dof","qiao_q2","unitree_g1", "booster_t1", "stanford_toddy"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=True,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--bvh_name",
        default='',
        type=str,
    )

    parser.add_argument(
        "--data_frames",
        default=-1,
        type=int,
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    bvh_file_name = [args.bvh_name+'.bvh']

    origin_root_pos=[1.6945454 , -3.135873,0]

    # motion_fps = 30
    # robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
    #                                     motion_fps=motion_fps,
    #                                     transparent_robot=0,
    #                                     record_video=args.record_video,
    #                                     video_path=args.video_path,
    #                                     # video_width=2080,
    #                                     # video_height=1170
    #                                     )

   
   
        
    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        if args.bvh_name != '':
            filenames = bvh_file_name

        print('++++++++++++',filenames)
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            
            # get the target file path
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            # Load LAFAN1 trajectory
            try:
                lafan1_data_frames, actual_human_height = load_lafan1_file(bvh_file_path)
                src_fps = 30  # LAFAN1 data is typically 30 FPS
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            
            # Initialize the retargeting system
            retarget = GMR(
                src_human="bvh",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retarget.xml_file)
            data = mj.MjData(model)

            

            # retarget to get all qpos
            qpos_list = []
            rot_vec_all = []
            if args.data_frames != -1:
                data_frames = args.data_frames 
            else:
                data_frames = len(lafan1_data_frames)

            for curr_frame in range(data_frames):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                qpos = retarget.retarget(smplx_data)
                
                qpos_list.append(qpos.copy())

                qpos_rot = qpos[3:7]
                qpos_rot[0],qpos_rot[1],qpos_rot[2],qpos_rot[3] = qpos_rot[1],qpos_rot[2],qpos_rot[3],qpos_rot[0]
                rotation = R.from_quat(qpos_rot)
                rotvec = rotation.as_rotvec()
                rotvec = torch.from_numpy(rotvec)
                
                rot_vec_all.append(rotvec)
            
            qpos_list = np.array(qpos_list)
            rot_vec_all = torch.cat(rot_vec_all, dim=0).view(-1, 3).float()

            # Initialize the forward kinematics
            device = "cuda:0"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            
            root_pos = qpos_list[:, :3] - origin_root_pos
            root_rot = qpos_list[:, 3:7]
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
            dof_pos = qpos_list[:, 7:]
            num_frames = root_pos.shape[0]
            
            # obtain local body pos
            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0
            local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos, 
                identity_root_rot, 
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
            body_names = kinematics_model.body_names

            HEIGHT_ADJUST = False
            PERFRAME_ADJUST = False
            if HEIGHT_ADJUST:
                body_pos, _ = kinematics_model.forward_kinematics(
                    torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                    torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
                )
                ground_offset = 0.00
                if not PERFRAME_ADJUST:
                    lowest_height = torch.min(body_pos[..., 2]).item()
                    root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
                else:
                    for i in range(root_pos.shape[0]):
                        lowest_body_part = torch.min(body_pos[i, :, 2])
                        root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

            N = rot_vec_all.shape[0]
            if args.robot == 'unitree_g1':
                pose_aa = torch.cat([rot_vec_all[None, :, None], G1_ROTATION_AXIS * dof_pos[None,:,:,None], torch.zeros((1, N, 3, 3))], axis = 2)
            if args.robot == 'qiao_q2':
                pose_aa = torch.cat([rot_vec_all[None, :, None], Q2_ROTATION_AXIS * dof_pos[None,:,:,None], torch.zeros((1, N, 3, 3))], axis = 2)
            if args.robot == 'qiao_q2_19dof':
                pose_aa = torch.cat([rot_vec_all[None, :, None], Q2_19dof_ROTATION_AXIS * dof_pos[None,:,:,None], torch.zeros((1, N, 3, 3))], axis = 2)
            
            with torch.no_grad():
                # print('++++++++++++++',pose_aa)
                feet_l , feet_r = foot_detect(pose_aa.squeeze().cpu().detach())  
                contact_mask = np.concatenate([feet_l,feet_r],axis=-1)
            
            data_name = args.robot + '_' + args.bvh_name
            motion_data = {}

            motion_data[data_name] = {
                "root_trans_offset": root_pos.astype(np.float32) ,
                "root_rot": root_rot.astype(np.float32) ,
                "pose_aa": pose_aa.squeeze().cpu().detach().numpy().astype(np.float32) , 
                "dof": dof_pos.astype(np.float32) ,
                # "local_body_pos": local_body_pos.detach().cpu().numpy(),
                "fps": src_fps,
                "contact_mask": contact_mask.astype(np.float32) ,
                # "link_body_list": body_names,
            }
            tgt_file_path = tgt_folder + args.robot + "/" + args.bvh_name + "_" + str(N) + ".pkl"
            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                joblib.dump(motion_data, tgt_file_path)

            



            # os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            # with open(tgt_file_path, "wb") as f:
            #     pickle.dump(motion_data, f)

    print("Done. saved to ", tgt_folder)
