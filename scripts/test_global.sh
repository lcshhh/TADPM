ENT="python second_stage.py "
cfg="config/ae_256.yaml"
#内部点事mask, 表面是mask_surface
ckpts="/data3/leics/dataset/checkpoints/voxel256/mask_surface/ckpt-last.pth"
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