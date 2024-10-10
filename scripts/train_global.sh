ENT="python second_stage.py "
cfg="config/vqvae.yaml"
saveroot="/data3/leics/dataset/checkpoints/vqvae"
ae_checkpoints="/data3/leics/dataset/teeth_full/single_checkpoints/pplus3/ckpt-best.pth"
exp_name="vqvae3_withoutvq"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ae_checkpoint $ae_checkpoints \
