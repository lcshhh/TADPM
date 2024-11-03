ENT="python second_stage.py "
cfg="config/ae.yaml"
saveroot="/data3/leics/dataset/checkpoints/voxel512"
#内部的名字叫mask,外部叫mask_surface
# ckpts="/data3/leics/dataset/checkpoints/voxel/mask_surface_64_128points/best.pth"
exp_name="mask_surface2"
launcher="none"
#mask_surface num_level=3
#mask_surface2 num_level=2



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    # --ckpts $ckpts
