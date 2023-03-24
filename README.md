# TPH-YOLOv5
This repo is the implementation of ["TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios"](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.html) and ["TPH-YOLOv5++: Boosting Object Detection on Drone-Captured Scenarios with Cross-Layer Asymmetric Transformer"](https://www.mdpi.com/2072-4292/15/6/1687).   
On [VisDrone Challenge 2021](http://aiskyeye.com/), TPH-YOLOv5 wins 4th place and achieves well-matched results with 1st place model.
![image](result.png)  
You can get [VisDrone-DET2021: The Vision Meets Drone Object Detection Challenge Results](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Cao_VisDrone-DET2021_The_Vision_Meets_Drone_Object_Detection_Challenge_Results_ICCVW_2021_paper.html) for more information. The TPH-YOLOv5++, as an improved version, significantly improves inference efficiency and reduces computational costs while maintaining detection performance compared to TPH-YOLOv5.

# Install
```bash
$ git clone https://github.com/cv516Buaa/tph-yolov5
$ cd tph-yolov5
$ pip install -r requirements.txt
```
# Convert labels
VisDrone2YOLO_lable.py transfer VisDrone annotiations to yolo labels.  
You should set the path of VisDrone dataset in VisDrone2YOLO_lable.py first.
```bash
$ python VisDrone2YOLO_lable.py
```

# Inference
* `Datasets` : [VisDrone](http://aiskyeye.com/download/object-detection-2/), [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5)
* `Weights` (PyTorch
v1.10): 
    * `yolov5l-xs-1.pt`:  | [Baidu Drive(pw: vibe)](https://pan.baidu.com/s/1APETgMoeCOvZi1GsBZERrg). |  [Google Drive](https://drive.google.com/file/d/1nGeKl3qOa26v3haGSDmLjeA0cjDD9p61/view?usp=sharing) |
    * `yolov5l-xs-2.pt`:  | [Baidu Drive(pw: vffz)](https://pan.baidu.com/s/19S84EevP86yJIvnv9KYXDA). |  [Google Drive](https://drive.google.com/file/d/1VmORvxNtvMVMvmY7cCwvp0BoL6L3RGiq/view?usp=sharing) |
    
val.py runs inference on VisDrone2019-DET-val, using weights trained with TPH-YOLOv5.  
(We provide two weights trained by two different models based on YOLOv5l.)

```bash
$ python val.py --weights ./weights/yolov5l-xs-1.pt --img 1996 --data ./data/VisDrone.yaml
                                    yolov5l-xs-2.pt
--augment --save-txt  --save-conf --task val --batch-size 8 --verbose --name v5l-xs
```
![image](./images/result_in_VisDrone.png)
Inference on UAVDT is similar and results of TPH-YOLOv5++ on UAVDT are as follow:
![image](./images/result_in_UAVDT.png)

# Ensemble
If you inference dataset with different models, then you can ensemble the result by weighted boxes fusion using wbf.py.  
You should set img path and txt path in wbf.py.
```bash
$ python wbf.py
```

# Train
train.py allows you to train new model from strach.
```bash
$ python train.py --img 1536 --adam --batch 4 --epochs 80 --data ./data/VisDrone.yaml --weights yolov5l.pt --hy data/hyps/hyp.VisDrone.yaml --cfg models/yolov5l-xs-tph.yaml --name v5l-xs-tph
$ python train.py --img 1536 --adam --batch 4 --epochs 80 --data ./data/VisDrone.yaml --weights yolov5l.pt --hy data/hyps/hyp.VisDrone.yaml --cfg models/yolov5l-tph-plus.yaml --name v5l-tph-plus
```
![image](train.png)  

# Description of TPH-YOLOv5, TPH-YOLOv5++ and citations
- https://arxiv.org/abs/2108.11539
- https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.html 
- https://www.mdpi.com/2072-4292/15/6/1687

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn or liubinghao@buaa.edu.cn  
If you find this code useful please cite:
```
@InProceedings{Zhu_2021_ICCV,
    author    = {Zhu, Xingkui and Lyu, Shuchang and Wang, Xu and Zhao, Qi},
    title     = {TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {2778-2788}
}

@Article{rs15061687,
   AUTHOR = {Zhao, Qi and Liu, Binghao and Lyu, Shuchang and Wang, Chunlei and Zhang, Hong},
   TITLE = {TPH-YOLOv5++: Boosting Object Detection on Drone-Captured Scenarios with Cross-Layer Asymmetric Transformer},
   JOURNAL = {Remote Sensing},
   VOLUME = {15},
   YEAR = {2023},
   NUMBER = {6},
   ARTICLE-NUMBER = {1687},
   URL = {https://www.mdpi.com/2072-4292/15/6/1687},
   ISSN = {2072-4292},
   DOI = {10.3390/rs15061687}
}
```

# References
Thanks to their great works
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
