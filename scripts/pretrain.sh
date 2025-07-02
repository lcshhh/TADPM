ENT="python pretrain.py "
cfg="config/pretrain.yaml"
saveroot="*"
exp_name="*"



$ENT --config $cfg \
    --save_root $saveroot \
    --exp_name $exp_name 

