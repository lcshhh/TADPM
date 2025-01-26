ENT="python third_stage.py "
cfg="config/DiffusionVAE_128.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion128"
exp_name="resolution"
launcher="none"
vae_checkpoint="/data3/leics/dataset/checkpoints/voxel128/resolution3/ckpt-best.pth"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --vae_checkpoint $vae_checkpoint \

