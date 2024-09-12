ENT="python second_stage.py "
cfg="config/GlobalVAE.yaml"
ckpts="/data3/leics/dataset/checkpoints/global/GlobalVAE/ckpt-best.pth"
saveroot="/data3/leics/dataset/checkpoints/global"
exp_name="GlobalVAE"
launcher="none"



$ENT --config $cfg \
    --launcher $launcher \
    --ckpts $ckpts \
    --save_root $saveroot \
    --exp_name $exp_name \
    --test