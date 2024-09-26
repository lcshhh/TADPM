ENT="python third_stage.py "
cfg="config/DiffusionVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion"
ckpts="/data3/leics/dataset/checkpoints/diffusion/DiffusionVAE4/ckpt-best.pth"
exp_name="DiffusionVAE2"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts \
    --test

