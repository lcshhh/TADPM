ENT="python first_stage.py "
cfg="config/single_PVCNN.yaml"
saveroot="/data3/leics/dataset/teeth_full/single_checkpoints"
exp_name="PVCNN"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name

