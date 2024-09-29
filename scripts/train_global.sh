ENT="python second_stage.py "
cfg="config/GlobalVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/GlobalVAE"
exp_name="rec512"
launcher="none"
PCN_checkpoint="/data3/leics/dataset/teeth_full/single_checkpoints/PCN/ckpt-best.pth"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
