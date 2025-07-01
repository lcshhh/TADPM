ENT="python pretrain.py "
cfg="config/pretrain.yaml"
saveroot="/data3/leics/dataset/checkpoints/MeshMAE"
exp_name="resume"
ckpts="/data/lcs/checkpoints/mesh/single_no_center/loss-0.0057-304.pkl"



$ENT --config $cfg \
    --save_root $saveroot \
    --exp_name $exp_name \
    --ckpts $ckpts \

