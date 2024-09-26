ENT="python predict_axis.py "
cfg="config/AxisPredictor.yaml"
saveroot="/data3/leics/dataset/checkpoints/axis"
ckpts="/data3/leics/dataset/checkpoints/axis/axis2/ckpt-best.pth"
exp_name="axis"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts \
    --test

