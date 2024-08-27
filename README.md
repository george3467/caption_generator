# Image Caption Generator

* Trained on Tensorflow 2.15.0 and Keras-NLP 0.12.1
* Note: This model requires further training

## Contents
* [Repository Files](#repository-files)
* [Model](#model)
* [Object Detector](#object-detector)
* [Caption Generator](#caption-generator)
* [Dataset](#dataset)
* [Inference](#inference)
* [Further Improvement](#further-improvement)


## Repository Files

* dataset.py - This file contains the script to download and organize the dataset.
* preprocess.py - This file contains functions to preprocess the images and prepare the labels.
* detector.py - This file contains the custom_retinanet model for detecting objects and its training and inference scripts.
* caption.py - This file contains the caption genrator model and the its training and inference scripts.
* sample_images - This folder contains 5 sample images that are used for inference.

Note: The model weights are not included since the files are large.

## Model

This model is trained to identify objects in a given image and a generate a caption for the image. The model consists of two parts: Object Detector and Caption Generator. The Detector part of the model detects individual objects in the image. The Caption Generator part of the model uses these detected objects and pretrained features of the image to generate a short caption.


## Object Detector

For the detector model, the architecture of the RetinaNet model was used with several modifications. The differences between this detector model and the RetinaNet model are described below:

* <u>Bounding Boxes</u>: <br>
The RetinaNet model predicts a bounding box for each object detected, however this detector model does not predict bounding boxes since it only needs to classify the detected objects.
  
* <u>Anchor Boxes</u>:<br>
The RetinaNet paper uses anchors of area 32<sup>2</sup>, 64<sup>2</sup>, 128<sup>2</sup>, 256<sup>2</sup>, 512<sup>2</sup>. Since the goal of this detector model is to only predict a few prominent objects, only the 64<sup>2</sup>, 256<sup>2</sup>, 512<sup>2</sup> anchor areas are used here.

* <u>Anchor Labels</u>:<br>
The RetinaNet paper assigns object labels for anchor boxes with iou > 0.5 and a background label for iou < 0.4. For 0.4 < iou < 0.5, the anchor box is ignored. <br>
This model assigns object labels for iou > 0.2 and a background label for iou < 0.2. In the classification loss function, the background_label is given a lower weight relative to the object labels.<br>
Note: "iou" = intersection over union

* <u>Feature Pyramid Network</u>:<br>
The RetinaNet paper's pyramid's input = [C3, C4, C5] layers of the ResNet-50 model and the output = [P3, P4, P5, P6, P7]. In this model, the pyramid's input is only the C5 layer and output = [P5, P6, P7]. Since the number of anchor boxes were reduced, the feature pyramid's output was reduced to match it.

* <u>Classification Subnet</u>:<br>
The RetinaNet paper uses 4 Convolution(filters=256, kernel=3) layers whereas this model uses only 3 of these layers.

* <u>Loss Function</u>:<br>
The RetinaNet paper uses a customized loss function called Focal Loss. In this detection model, the Sparse Categorical CrossEntropy loss function is used. Here, the background label is given a weight of 0.1 and the object labels are given a weight of 1.

Reference to the RetinaNet paper:
```Bibtex
@misc{lin2018focallossdenseobject,
      title={Focal Loss for Dense Object Detection}, 
      author={Tsung-Yi Lin and Priya Goyal and Ross Girshick and Kaiming He and Piotr Dollár},
      year={2018},
      eprint={1708.02002},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1708.02002}, 
}
```

<br>This detector model also includes ideas from the following Keras Tutorial which contains an implementation of the original RetinaNet Model:
```Bibtex
https://keras.io/examples/vision/retinanet/
```


## Caption Generator

The Caption generator model uses the following steps:

1. An encoder transformer, "image_encoder", is applied to the C5 layer (as defined in RetinaNet) of the pretrained ResNet-50 model.

2. A second encoder transformer, "object_encoder", is applied to the list of objects detected in the image by the detector model.

3. A decoder transformer, "decoder_layer_1", takes the current sequence of words as the "decoder_sequence" input and the output of "image_encoder" as the "encoder_sequence" input.

4. A second decoder transformer, "decoder_layer_2", takes the output of "decoder_layer_1" as the "decoder_sequence" input and the output of "object_encoder" as the "encoder_sequence" input. This "decoder_layer_2" predicts the next word in the sequence.

<br>This caption generator model also includes ideas from the following Keras Tutorial:
```Bibtex
https://keras.io/examples/vision/image_captioning/
```


## Dataset

To train this model, the COCO dataset was used. Here, the Coco "2017 Validation" dataset was used instead of the training dataset since the validation dataset is smaller. This dataset contains 5,000 images. Only 800 of these images were used to train this model.

Reference to the Coco dataset:

```Bibtex
@misc{lin2015microsoftcococommonobjects,
      title={Microsoft COCO: Common Objects in Context}, 
      author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
      year={2015},
      eprint={1405.0312},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1405.0312}, 
}
```


## Inference

Here, 5 images from the dataset are used as samples. These 5 images were not used in training the model.

In some cases, the model predicts a sensible result as in Example 1.

### Example 1:

<img src="sample_images/000000090108.jpg" width="200">

Objects Detected - sink, toilet <br>
Caption Generated - a clean bathroom has a tub and sink<br><br>

---

In a lot of cases, the model's predicted caption is not completely accurate as in examples 2, 3 & 4.

### Example 2:

<img src="sample_images/000000086220.jpg" width="200">

Objects Detected - bus, train <br>
Caption Generated - a view of a public transportation truck on a city street

### Example 3:

<img src="sample_images/000000361586.jpg" width="200">

Objects Detected - person, handbag <br>
Caption Generated - a group of people standing around a room

### Example 4:

<img src="sample_images/000000367680.jpg" width="200">

Objects Detected - car, person <br>
Caption Generated - a bunch of cars are standing next to a lake <br><br>

---

In a few cases, the model's predicted caption is inaccurate as in Example 5.

### Example 5:

<img src="sample_images/000000472375.jpg" width="200">

Objects Detected - dog, couch <br>
Caption Generated - a cat sitting on a chair looking at something to in the foreground


## Further Improvement

This model's performance could potentially be improved by:

* using a larger number of images for training

* increasing the number of heads in the transformer layers

* increasing the intermediate dimensions of the transformer layers





