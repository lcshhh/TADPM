ENT="python first_stage.py "
cfg="config/PCN.yaml"
saveroot="/data3/leics/dataset/teeth_full/single_checkpoints"
exp_name="PCN"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name

