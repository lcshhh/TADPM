ENT="python third_stage.py "
cfg="config/DiffusionVAE_128.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion"
#内部是diffusionmask1e-3，外部是diffusionmask_surface
ckpts="/data3/leics/dataset/checkpoints/diffusion128/diffusionmask_surface/ckpt-best.pth"
exp_name="DiffusionVAE"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts \
    --test

