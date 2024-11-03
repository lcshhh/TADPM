ENT="python third_stage.py "
cfg="config/DiffusionVAE_256.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion256"
exp_name="diffusionmask_surface"
launcher="none"
vae_checkpoint="/data3/leics/dataset/checkpoints/voxel256/mask_surface/ckpt-last.pth"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --vae_checkpoint $vae_checkpoint \

