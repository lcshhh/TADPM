ENT="python third_stage.py "
cfg="config/DiffusionVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion"
exp_name="diffusionmask_surface"
launcher="none"
vae_checkpoint="/data3/leics/dataset/checkpoints/voxel/mask_surface/ckpt-best.pth"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --vae_checkpoint $vae_checkpoint \

