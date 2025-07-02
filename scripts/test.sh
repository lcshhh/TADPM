ENT="python main.py "
cfg="config/TADPM.yaml"
saveroot="*"
ckpts="*"
exp_name="*"



$ENT --config $cfg \
    --save_root $saveroot \
    --exp_name $exp_name \
    --encoder_ckpts $encoder_ckpts \
    --ckpts $ckpts \
    --test

