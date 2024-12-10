# ViPOcc Demo
python eval.py -cn demo \
  'model_conf.backbone.use_depth_branch=true' \
  'model_conf.draw_occ=true' \
  'model_conf.draw_rendered_depth=true' \
  'model_conf.draw_pred_depth=true' \
  'model_conf.draw_pseudo_depth=true' \
  'model_conf.draw_bev=true' \
  'model_conf.save_dir="visualization/vipocc"'
