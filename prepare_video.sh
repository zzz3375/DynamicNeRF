ROOT_PATH=~/DynamicNeRF
DATASET_NAME=DJI_20250516151729_0005_V
DATASET_PATH=$ROOT_PATH/data/$DATASET_NAME

mkdir -p $ROOT_PATH/weights
cd $ROOT_PATH/weights
# wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.pt


cd $ROOT_PATH/utils
python generate_data.py --videopath ../DJI_20250516151729_0005_V.MP4

colmap feature_extractor \
--database_path $DATASET_PATH/database.db \
--image_path $DATASET_PATH/images_colmap \
--ImageReader.mask_path $DATASET_PATH/background_mask \
--ImageReader.single_camera 1

colmap sequential_matcher \
--database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images_colmap \
    --output_path $DATASET_PATH/sparse \
    # --Mapper.num_threads 16 \
    # --Mapper.init_min_tri_angle 4 \
    # --Mapper.multiple_models 0 \
    # --Mapper.extract_colors 0

cd $ROOT_PATH/utils
python generate_pose.py --dataset_path $DATASET_PATH

cd $ROOT_PATH/utils
python generate_depth.py --dataset_path $DATASET_PATH --model $ROOT_PATH/weights/model-f6b98070.pt

cd $ROOT_PATH/utils
python generate_flow.py --dataset_path $DATASET_PATH --model $ROOT_PATH/weights/raft-things.pth

cd $ROOT_PATH/utils
python generate_motion_mask.py --dataset_path $DATASET_PATH

cd $ROOT_PATH/
python run_nerf.py --config configs/config-WTB-Beijing.txt