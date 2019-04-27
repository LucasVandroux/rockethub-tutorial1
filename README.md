# PyTorch Rocket Yolov3 RetinaNet SSD - Tutorial 2: A Tale of 3 Rockets
    Have you ever wanted to test multiple Deep Learning models and compare their results very easily?

    Are you tired of picking a Deep Learning model just because it is the only one you are able to run?

We want to solve this problem and we packaged 3 state-of-the-art Deep Learning models for Object detection for you to easily test them.

We are calling those new way of packaging Deep Learning models: __Rockets__.

__Welcome to the Rockets Scientists Community!!!__

## Install the repositories
We recommend you to use an isolated Python environement such as [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.conda.io/en/latest/) with at least __Python 3.5__. Then you can use the following lines of code:
```
git clone https://github.com/LucasVandroux/PyTorch-Rocket-YOLOv3-RetinaNet-SSD
cd PyTorch-Rocket-YOLOv3-RetinaNet-SSD
pip install -r requirements.txt
```
### Install PyTorch
As the installation for PyTorch is different for each platform, you need to look at the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Don't worry it is very simple, maximum 2 lines of codes :stuck_out_tongue_closed_eyes:

## A Tale of 3 Rockets
For this first tutorial, we selected three state-of-the-art models in Object Detection for you to play with:
1. SSD: Single Shot MultiBox Detector ___[[paper]](https://arxiv.org/pdf/1512.02325.pdf)___
2. RetinaNet ___[[paper]](https://arxiv.org/pdf/1708.02002.pdf)___
3. YOLOv3 ___[[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)___

## Run the Object Detection model
Everything is happening in the `detect.py` file. There you can choose which image and model to use with just one line of code.

Once you are ready you just need to run `python detect.py` and everything will happen magically.

Don't hesitate to play around by swapping the different Rockets and comparing their output.

To run the three models against all sample images, simply use:

```
python ./detect.py street lucas/ssd
python ./detect.py street igor/retinanet
python ./detect.py street lucas/yolov3
python ./detect.py office lucas/ssd
python ./detect.py office igor/retinanet
python ./detect.py office lucas/yolov3
python ./detect.py shop lucas/ssd
python ./detect.py shop igor/retinanet
python ./detect.py shop lucas/yolov3
```

## Outputs of the different Rockets
| Filename | Original | SSD | RetinaNet | YOLOv3 | Google Vision AI |
|----------|----------|-----------|-----|--------|------------------|
| `office.jpg` | ![image-original-office](images/office.jpg) | ![image-ssd-office](images/detections/ssd/office.jpg) | ![image-retinanet-office](images/detections/retinanet/office.jpg)|![image-yolov3-office](images/detections/yolov3/office.jpg)|![image-googleAPI-office](images/detections/googleAPI/office.jpg)|
|`shop.jpg`|![image-original-shop](images/shop.jpg)|![image-ssd-shop](images/detections/ssd/shop.jpg)|![image-retinanet-shop](images/detections/retinanet/shop.jpg)|![image-yolov3-shop](images/detections/yolov3/shop.jpg)|![image-googleAPI-shop](images/detections/googleAPI/shop.jpg)|
|`street.jpg`|![image-original-street](images/street.jpg)|![image-ssd-street](images/detections/ssd/street.jpg)|![image-retinanet-street](images/detections/retinanet/street.jpg)|![image-yolov3-street](images/detections/yolov3/street.jpg)|![image-googleAPI-street](images/detections/googleAPI/street.jpg)|

We added the outputs from the [Google Vision AI](https://cloud.google.com/vision/) to compare with the results of our Rockets.

The Rockets are also outputting a Json formatted answer that you can use to integrate the Rockets in one of your Kickass project.

## Contact
Any feedback or complaint from your neighbors about the noise your Rockets are making, please contact us at [hello@mirage.id](mailto:hello@mirage.id). 
