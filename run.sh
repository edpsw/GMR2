# single motion
python scripts/bvh_to_robot.py --bvh_file /home/z/code/ubisoft-laforge-animation-dataset/output/BVH/dance1_subject2.bvh \
--robot engineai_pm01  \
--save_path output

/home/z/code/GMR/output/g1_dance1_subject2.pkl

# unitree_g1', 'booster_t1', 'stanford_toddy', 'fourier_n1', 'engineai_pm01

python scripts/bvh_to_robot_dataset.py \
--src_folder /home/z/code/ubisoft-laforge-animation-dataset/output/BVH \
--robot unitree_g1    \
--bvh_name dance1_subject2 

# --tgt_folder <path_to_dir_to_save_robot_data> 


python scripts/vis_robot_motion.py \
--robot unitree_g1 \
--robot_motion_path <path_to_save_robot_data.pkl>




###########q2 19dof

python scripts/bvh_to_robot.py --bvh_file /home/z/code/ubisoft-laforge-animation-dataset/output/BVH/dance1_subject2.bvh \
--robot qiao_q2_19dof  \


python scripts/bvh_to_robot_dataset.py \
--src_folder <path_to_dir_of_bvh_data> \
--tgt_folder <path_to_dir_to_save_robot_data> --robot <robot_name>



python scripts/bvh_to_robot_asap.py  \
--robot qiao_q2_19dof \
--src_folder /home/z/code/ubisoft-laforge-animation-dataset/output/BVH \
--bvh_name dance1_subject2 \
--data_frames 1200





######################################################

cd path/to/GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s

python tools/demo/demo.py --video=/home/z/Downloads/taiji.mp4 -s

python tools/demo/demo.py --video=/home/z/Downloads/jile.mp4 -s


在 GVHMR/outputs/demo/tennis/hmr4d_results.pt 中获取已保存的人体姿势数据。

python scripts/gvhmr_to_robot.py \
--gvhmr_pred_file /home/z/code/gvhmr/outputs/demo/tennis/hmr4d_results.pt \
--robot unitree_g1 \
--record_video \
--save_path motion_data/video/g1/tennis.pkl



python scripts/gvhmr_to_robot.py \
--gvhmr_pred_file /home/z/code/gvhmr/outputs/demo/tennis/hmr4d_results.pt \
--robot qiao_q2_19dof \
--record_video \
--save_path motion_data/video/q2/tennis.pkl

python scripts/gvhmr_to_robot.py \
--gvhmr_pred_file /home/z/code/gvhmr/outputs/demo/taiji/hmr4d_results.pt \
--robot qiao_q2_19dof \
--record_video \
--save_path motion_data/video/q2/taiji.pkl

python scripts/gvhmr_to_robot.py \
--gvhmr_pred_file /home/z/code/gvhmr/outputs/demo/jile/hmr4d_results.pt \
--robot qiao_q2_19dof \
--record_video \
--save_path motion_data/video/q2/jile.pkl




python scripts/gvhmr_to_robot.py --gvhmr_pred_file /home/z/code/gvhmr/outputs/demo/2795750-uhd_3840_2160_25fps/2795750-uhd_3840_2160_25fps_3_incam_global_horiz.mp4 --robot unitree_g1 --record_video