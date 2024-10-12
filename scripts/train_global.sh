ENT="python second_stage.py "
cfg="config/ae.yaml"
saveroot="/data3/leics/dataset/checkpoints/voxel"
exp_name="residual"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
