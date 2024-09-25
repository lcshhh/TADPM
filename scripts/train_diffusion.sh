ENT="python third_stage.py "
cfg="config/DiffusionVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/diffusion"
exp_name="DiffusionVAE2"
launcher="none"
PCN_checkpoint="/data3/leics/dataset/teeth_full/single_checkpoints/PCN/ckpt-best.pth"
vae_ckpts="/data3/leics/dataset/checkpoints/global_axis5/GlobalVAE/ckpt-best.pth"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --vae_ckpts $vae_ckpts \
    --resume

