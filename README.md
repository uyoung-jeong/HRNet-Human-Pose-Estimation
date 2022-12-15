# Deep High-Resolution Representation Learning for Human Pose Estimation

## Note
This repo is cloned from the official HRNet repo.
We have made several modifications and might be unstable.
If you want to get information about the baseline model, please visit [official HRNet repo](https://github.com/HRNet/HRNet-Human-Pose-Estimation).
We only describe what we have changed.

## Environment
The code is developed using python 3.7 on Ubuntu 18.04. The code is developed and tested using 2 RTX TITAN GPU cards.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools
   ├── README.md
   └── requirements.txt
   ```

5. If you are using MLV lab's internal server, you do not have to download pretrained data or prepare dataset. Simple soft-link as following:
```
ln -s /syn_mnt/uyoung/human/HRNet/models
ln -s /syn_mnt/uyoung/human/HRNet/data
```

### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))


```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))


```
python tools/test.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
```

#### Training with occlusion augmentation on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_occ.yaml \

```

### Demo Commands
​```
--cfg = configuration file of HRNet
--fileType = Types of file. if video file, type 'vid', if image files 'img'.
--jsonDir = detection json file directory (['objects'][i]['h','w','y','x'] -> bbox)
--imagesDirectory = Directory address of the image.
--videoFile = Address of the video file on the hard disk.
--outputDir = Directory where you want to save results.
--inferenceFps = In case of video, how many frames/sec you want,
        default is 20.
--showImages = Type "True" if you want to show the images on the
        screen while running. Default is "False". No need to input if you
        dont want to show images in the screen.
--cudnnBenchmark = Type "True" if you want to run the demo with
        cudnn.BENCHMARK. Default is "False" because it takes a too much
        memory if the input images are not of same size. No need to input if you
        dont want to use it with cudnn.BENCHMARK
​```
#### Example Command for video
    python tools/demo.py --fileType vid --videoFile demo_samples/videos/basketball.mp4 --outputDir demo/videos/basktball/ --inferenceFps 15 --showImages False --cudnnBenchmark False --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth
​
#### Example Command for images
    python tools/demo.py --fileType img --imagesDirectory data/coco/images/val2017/ --outputDir demo/coco_val  --showImages True --cudnnBenchmark False --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
