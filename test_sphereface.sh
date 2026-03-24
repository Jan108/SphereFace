#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/SphereFace'
DATA_DIR='/mnt/data/afarec/data/PetFace'

for backbone in "64"; do
  for cls in "bird" "cat" "dog" "small_animals"; do
    if [ $backbone == "20" ]; then
        continue
    fi
    echo "Start evaluation for SphereFace with SFNet${backbone} for class ${cls}"
    PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
      CUDA_VISIBLE_DEVICES=0 \
      python $ROOT_DIR/evaluation.py \
        --proj_dir "${ROOT_DIR}/work_dir/${backbone}_${cls}" \
        --img_path "${DATA_DIR}/images" \
        --img_verification "${DATA_DIR}/split/${cls}/verification.csv" \
        --img_identification "${DATA_DIR}/split/${cls}/identification_img.csv"
  done
  echo "Start evaluation for SphereFace with SFNet${backbone} for class all"
  PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=0 \
  python $ROOT_DIR/evaluation.py \
    --proj_dir "${ROOT_DIR}/work_dir/${backbone}_all" \
    --img_path "${DATA_DIR}/images" \
    --img_verification "${DATA_DIR}/split/all/verification.csv" \
    --img_identification "${DATA_DIR}/split" \
    --ident-general
done
