ROOT=/data3/leics/dataset/type2_process_final

DATAROOT=$ROOT/single_before
MANIFOLD=$ROOT/manifold_before
SIMPLIFY=$ROOT/simplify_before
REMESH=$ROOT/remesh_before

python data_preprocess/manifold.py \
    --dataroot "$DATAROOT" \
    --manifold "$MANIFOLD" \
    --simplify "$SIMPLIFY"

python data_preprocess/simplify.py \
    --dataroot "$DATAROOT" \
    --manifold "$MANIFOLD" \
    --simplify "$SIMPLIFY"

python data_preprocess/datagen_maps.py \
    --simplify "$SIMPLIFY" \
    --output "$REMESH"