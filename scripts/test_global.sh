ENT="python second_stage.py "
cfg="config/GlobalVAE.yaml"
ckpts="/data3/leics/dataset/checkpoints/global/GlobalVAE_axis/ckpt-best.pth"
saveroot="/data3/leics/dataset/checkpoints/global"
exp_name="GlobalVAE_axis"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --ckpts $ckpts \
    --save_root $saveroot \
    --exp_name $exp_name \
    --test