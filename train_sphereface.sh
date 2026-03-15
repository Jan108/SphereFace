#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/SphereFace'

for backbone in "20" "64"; do
  for cls in "all" "bird" "cat" "dog" "small_animals"; do
    echo "Start training for SphereFace with SFNet${backbone} for class ${cls}"
    PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
      CUDA_VISIBLE_DEVICES=0 \
      python $ROOT_DIR/main_train.py \
        --cfg_path "${ROOT_DIR}/config/afarec/sfnet${backbone}_${cls}.yml" \
        --proj_dir "${ROOT_DIR}/work_dir/${backbone}_${cls}"
  done
done
