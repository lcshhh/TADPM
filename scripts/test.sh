ENT="python first_stage.py "
cfg="config/PCN.yaml"
saveroot="/data3/leics/dataset/teeth_full/single_checkpoints"
ckpts="/data3/leics/dataset/teeth_full/single_checkpoints/PCN/ckpt-best.pth"
exp_name="PCN"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --ckpts $ckpts \
    --save_root $saveroot \
    --exp_name $exp_name \
    --test
