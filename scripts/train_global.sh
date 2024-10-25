ENT="python second_stage.py "
cfg="config/ae.yaml"
saveroot="/data3/leics/dataset/checkpoints/voxel"
#内部的名字叫mask,外部叫mask_surface
exp_name="mask_surface2"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
