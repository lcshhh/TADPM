ENT="python third_stage.py "
cfg="config/DiffusionVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion"
ckpts="/data3/leics/dataset/checkpoints/diffusion/DiffusionVAE_newloss/ckpt-last.pth"
exp_name="DiffusionVAE"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts \
    --test

