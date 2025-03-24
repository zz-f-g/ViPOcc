# ====================================== ViPOcc Evaluation ======================================
# eval occ
python eval.py -cn eval_lidar_occ
#scene_O_acc    : 0.927
#scene_IE_acc   : 0.713
#scene_IE_rec   : 0.686
#object_O_acc   : 0.790
#object_IE_acc  : 0.685
#object_IE_rec  : 0.643


# eval rendered depth
python eval.py -cn eval_depth_kitti360
#abs_rel        : 0.097
#sq_rel         : 0.446
#rmse           : 3.383
#rmse_log       : 0.188
#a1             : 0.886
#a2             : 0.956
#a3             : 0.980


# eval predicted depth
python eval.py -cn eval_depth_kitti360 \
  "model_conf.eval_rendered_depth=false"
#abs_rel        : 0.097
#sq_rel         : 0.441
#rmse           : 3.427
#rmse_log       : 0.188
#a1             : 0.883
#a2             : 0.957
#a3             : 0.980


# ========================== BTS Official Evaluation ==========================
# KITTI-360, occupancy prediction
python eval.py -cn eval_lidar_occ \
"checkpoint=\"checkpoints/BTS-KITTI-360.pt\"" \
'model_conf.z_range=[20, 4]' \
'model_conf.backbone.use_depth_branch=false'
#scene_O_acc    : 0.923
#scene_IE_acc   : 0.690
#scene_IE_rec   : 0.644
#object_O_acc   : 0.791
#object_IE_acc  : 0.689
#object_IE_rec  : 0.601


# KITTI-360, depth estimation
python eval.py -cn eval_depth_kitti360 \
"checkpoint=\"checkpoints/BTS-KITTI-360.pt\""
#abs_rel        : 0.103
#sq_rel         : 0.492
#rmse           : 3.007
#rmse_log       : 0.194
#a1             : 0.891
#a2             : 0.958
#a3             : 0.978



# KITTI-360, pseudo depth estimation (median scaling)
python eval.py -cn eval_pseudo_depth
#abs_rel        : 0.142
#sq_rel         : 0.533
#rmse           : 3.297
#rmse_log       : 0.209
#a1             : 0.832
#a2             : 0.952
#a3             : 0.981


# (no scaling)
python eval.py -cn eval_pseudo_depth \
'model_conf.depth_scaling=null'
#abs_rel        : 0.586
#sq_rel         : 3.743
#rmse           : 6.653
#rmse_log       : 0.477
#a1             : 0.071
#a2             : 0.598
#a3             : 0.889



python eval.py -cn eval_depth_kitti360 "model_conf.mlp_fine.type=empty"
python eval.py -cn eval_depth_kitti360 "checkpoint=outputs/2025-03-17/14-36-58/kitti_360/best_model_24_abs_rel\=-0.1042.pt" 
python eval.py -cn eval_depth_kitti360 "checkpoint=outputs/2025-03-17/14-36-58/kitti_360/best_model_24_abs_rel\=-0.1042.pt" "model_conf.eval_teacher=false"
