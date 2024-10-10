ENT="python first_stage.py "
cfg="config/PCN.yaml"
saveroot="/data3/leics/dataset/teeth_full/single_checkpoints"
ckpts="/data3/leics/dataset/teeth_full/single_checkpoints/PCN512/ckpt-best.pth"
exp_name="PCN2_512"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts

