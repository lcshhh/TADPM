ENT="python third_stage.py "
cfg="config/DiffusionVAE_256.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion"
#内部是diffusionmask1e-3，外部是diffusionmask_surface
ckpts="/data3/leics/dataset/checkpoints/diffusion256/diffusionmask_surface_newloss/ckpt-best.pth"
exp_name="DiffusionVAE"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts \
    --test

