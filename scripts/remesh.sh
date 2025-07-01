python data_preprocess/manifold.py \
    --dataroot '/data3/leics/dataset/type2_process_final/single_before' \
    --manifold '/data3/leics/dataset/type2_process_final/manifold_before' \
    --simplify '/data3/leics/dataset/type2_process_final/simplify_before' 

# dataroot:The root to single tooth before treatment
# manifold: Output path for manifold

python data_preprocess/simplify.py \
    --dataroot '/data3/leics/dataset/type2_process_final/single_before' \
    --manifold '/data3/leics/dataset/type2_process_final/manifold_before' \
    --simplify '/data3/leics/dataset/type2_process_final/simplify_before' 

# simplify: Output path for simplify

python data_preprocess/datagen_maps.py \
    --dataroot '/data3/leics/dataset/type2_process_final/simplify_before' \
    --output '/data3/leics/dataset/type2_process_final/remesh_before' \

# dataroot：Ther root for single tooth after simplification
# output: Output root for data after remesh

