ENT="python second_stage.py "
cfg="config/ae.yaml"
ckpts="/data3/leics/dataset/checkpoints/voxel/residual/ckpt-best.pth"
saveroot="/data3/leics/dataset/checkpoints/global"
exp_name="vqvae"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --ckpts $ckpts \
    --save_root $saveroot \
    --exp_name $exp_name \
    --test

#global_axis4里面axis为什么是正常的？