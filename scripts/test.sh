ENT="python main.py "
cfg="config/TADPM.yaml"
saveroot="/data3/leics/dataset/checkpoints/tadpm_new"
encoder_ckpts="/data3/leics/dataset/checkpoints/MeshMAE/resume/ckpt-best.pth"
ckpts="/data3/leics/dataset/checkpoints/tadpm_new/tadpm_resume/ckpt-best.pth"
exp_name="mlp"



$ENT --config $cfg \
    --save_root $saveroot \
    --exp_name $exp_name \
    --encoder_ckpts $encoder_ckpts \
    --ckpts $ckpts \
    --test

