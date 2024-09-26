ENT="python second_stage.py "
cfg="config/GlobalVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/global_axis"
exp_name="GlobalVAE_type2_4"
launcher="none"
ckpts="/data3/leics/dataset/checkpoints/global_axis/GlobalVAE_type2_3/ckpt-best.pth"
PCN_checkpoint="/data3/leics/dataset/teeth_full/single_checkpoints/PCN/ckpt-best.pth"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts

