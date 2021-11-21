#!/bin/bash
x=1
while [ $x -le 4 ]
do
  echo "Training for -12 OCC_MIN_JOINT, 1 OCC_HIDE_NUM: $x th times"
  python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_occ_12_1.yaml
  x=$(( $x + 1 ))
done

x=1
while [ $x -le 4 ]
do
  echo "Training for -10 OCC_MIN_JOINT, 2 OCC_HIDE_NUM: $x th times"
  python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_occ_10_2.yaml
  x=$(( $x + 1 ))
done

echo "Expreiment Finished!"
