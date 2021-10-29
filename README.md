# Deep High-Resolution Representation Learning for Human Pose Estimation (accepted to CVPR2019)
# This contains demo code of HRNet.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Commands
     * --cfg = configuration file of HRNet
     * --fileType = Types of file. if video file, type 'vid', if image files 'img'.
     * --imagesDirectory = Directory address of the image.
     * --videoFile = Address of the video file on the hard disk.
     * --outputDir = Directory where you want to save results.
     * --inferenceFps = In case of video, how many frames/sec you want, 
        default is 20.
     * --showImages = Type "True" if you want to show the images on the
        screen while running. Default is "False". No need to input if you 
        dont want to show images in the screen.
     * --cudnnBenchmark = Type "True" if you want to run the demo with 
        cudnn.BENCHMARK. Default is "False" because it takes a too much
        memory if the input images are not of same size. No need to input if you 
        dont want to use it with cudnn.BENCHMARK

## Example Command for video
    python tools/demo.py --fileType vid --videoFile demo_samples/videos/basketball.mp4 --outputDir demo/videos/basktball/ --inferenceFps 15 --showImages False --cudnnBenchmark False --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth 

## Example Command for images
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
