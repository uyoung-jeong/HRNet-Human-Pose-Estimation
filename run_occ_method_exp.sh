#!/bin/bash
for ((x=1; x<=4; x++))
do
  echo "Training baseline with 40 epochs: $x th times"
  python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_occ_base.yaml
done

for ((x=1; x<=4; x++))
do
  echo "Training for -12 OCC_MIN_JOINT, 1 OCC_HIDE_NUM, anchor: $x th times"
  python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_occ_12_1_anchor.yaml
done

for ((x=1; x<=4; x++))
do
  echo "Training for -12 OCC_MIN_JOINT, 1 OCC_HIDE_NUM, random: $x th times"
  python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_occ_12_1_random.yaml
done

echo "Expreiment Finished!"
