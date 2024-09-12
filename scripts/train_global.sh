ENT="python second_stage.py "
cfg="config/GlobalVAE.yaml"
saveroot="/data3/leics/dataset/checkpoints/global"
exp_name="GlobalVAE_axis2"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name

