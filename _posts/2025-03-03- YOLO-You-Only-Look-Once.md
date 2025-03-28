---
layout: post
title: "YOLO - YOU ONLY LOOK ONCE - Real-Time Object Detection"
date: 2025-03-03
categories: [statistics, data-science, machine-learning, education]
---

I first came across YOLO in a computer vision article. The clickbaity title immediately caught my attention, and I looked it up online. It's a brilliant paper from researchers at the University of Washington (links below) written in plain English - "YOU ONLY LOOK ONCE: Unified, Real-Time Object Detection." I thought of sharing the research discussed in this paper as I was amazed by the simplicity of the idea and its implications for computer vision.

## Before jumping into YOLO - I think it's important to give some context

YOLO is great for rapid object detection - think Autonomous Cars, Robots, etc.

Our human mind is just amazing. We look at an object and instantly recognize what it is, its location within the environment, and how it is interacting within that environment. Based on those inputs, we act accordingly. For example, when we are driving and we see a red light, a pedestrian crossing, or a large object on our driving path, we instantly know how to react.

It turns out that getting cars or robots to do the same takes a lot of hardware - cameras, LiDARs, powerful microprocessors for processing and analyzing information - and software - complex algorithms for detecting and identifying objects (perception), determining where the car is (localization), predicting what others will do, planning the car's path, and executing that plan through control systems.

YOLO fits into the identifying objects (perception) bucket - it takes 2D images from cameras as inputs and classifies objects in the outputs - e.g., pedestrian, bus, cycle, dog, etc.

There are several other algorithms like R-CNN, DPM, Deep Multi-Box, OverFeat, MultiGrasp, etc. for OBJECT DETECTION. (I sometimes wonder where researchers get these names from).

For example, Tesla FSD uses HydraNet for object detection, which is inspired by YOLO's philosophy, whereas Waymo uses a 2-stage detection: Stage 1 - LiDAR point cloud processed by VoxelNet, and Stage 2 - Cameras for refined detections (similar to R-CNN).

## Let's get back to YOLO

Simply put, YOLO looks at the big picture and reasons globally, whereas traditional R-CNNs look at the details piece by piece.

The TRAINING workflow is: Input image + labeled bounding boxes → a CNN extracts features → a final layer predicts → Loss function trains the CNN to minimize errors.

The input image is divided into a grid. Let's take an example of an image of a dog. The image (100×100 pixels) is first divided into a grid (say 3×3, i.e., each cell is approximately 33×33 pixels).

Let's say the dog's center is at (50, 50), hence it falls in cell (1, 1) (i.e., pixels 33,33 to 66,66). Note we are just talking about the dog's center. Parts of the dog might fall into other cells. Say the total size is w = 40 pixels and h = 60 pixels.

For cell (1, 1), the dog's absolute coordinates are converted into YOLO FORMAT: (x_cell, y_cell, w, h)

- x_cell = (50-33)/33 = 0.5 → Position of the dog's center within its assigned grid cell
- y_cell = (50-33)/33 = 0.5 → Position of the dog's center within its assigned grid cell
- w = 40/100 = 0.4 (dog's width = 40 pixels) → Width of the bounding box as a fraction of the entire image
- h = 60/100 = 0.6 (dog's height = 60 pixels) → Height of the bounding box as a fraction of the entire image

The dog's width is 40% of the image width, and its height is 60% of the image height.

![YOLO Grid Cell Prediction](/images/yolo-grid-visualization.svg)
*How a single grid cell in YOLO predicts the entire bounding box of an object, even when the object extends beyond the cell boundaries.*

Why do we need to calculate these relative values? It works for any image size and simplifies learning for the CNN.

Now we need to create the Ground Truth Label for cell (1, 1):

The format is [x_cell=0.5, y_cell=0.5, w=0.4, h=0.6, confidence=1.0, class="dog"]

- Relative center: (0.5, 0.5)
- Relative size: (0.4, 0.6)
- Confidence: 1.0 (since the dog exists in this cell)
- Class probabilities: [1.0, 0.0] (assuming binary classes: "dog" or "cat" - just to make our life easy)

Confused? This might help: We are in the training phase. We need to train the CNN to recognize dogs, cats, humans, etc. How can you train it? By providing it with labeled data so that it can learn. To simplify, a human annotator draws a box around each object (e.g., a dog) in the image, records the class ("dog"), and saves the absolute coordinates of the box. Conversion to YOLO format (all the math we did above) is done automatically using software tools. The output will be a ground truth label as shown above.

## Now we have an image of a dog (100×100) and a ground truth label. What's next?

If you don't want to read the details, here's the workflow: Image → CNN → Raw Prediction → Compare to Ground Truth → Calculate Loss → Update Weights → Repeat

The next step is to train the model on this picture. The image along with the label is the input to the CNN. The CNN's job is to learn to predict the dog's box and class from scratch, adjusting its weights to match the ground truth.

Let's break it down: the input image passes through the layers of the CNN (Convolution, Pooling, etc. - think of it as a black box). Early layers detect edges/colors → later layers detect shapes → final layers detect objects.

The output will be a prediction for cell [1, 1] which might look something like [x_cell=0.3, y_cell=0.7, w=0.2, h=0.3, confidence=0.5, class="dog":0.6]

The predicted output is compared with the labeled ground truth data to calculate the loss as a summarized single number (e.g., 0.85):
- Box Center: Off by (0.3-0.5=-0.2, 0.7-0.5=+0.2)
- Box Size: Too small (0.2 vs 0.4 width, 0.3 vs 0.6 height)
- Confidence: Too low (0.5 vs 1.0)
- Class: Only 60% sure it's a dog (should be 100%)

The CNN calculates how much each weight contributed to the error, then an optimizer tweaks the weights to reduce the error.

As you can imagine, the initial guesses will be random and error-prone. As we keep training the CNN, the cell (1, 1) prediction will get closer to the ground truth, eventually reaching something like [x_cell=0.49, y_cell=0.51, w=0.39, h=0.61, confidence=0.98, class="dog":0.99]

The models are trained on millions of images, and once the training process begins, the same images are used multiple times but in different orders.

Here is an example of training on 3 Images (2 Epochs):

**Epoch 1:**
- Image A (dog) → Predict → Update weights
- Image B (cat) → Predict → Update weights
- Image C (car) → Predict → Update weights

**Epoch 2 (shuffled):**
- Image B (cat) → Predict → Update weights
- Image C (car) → Predict → Update weights
- Image A (dog) → Predict → Update weights

The dog image is reused, but the model sees it mixed with other images to avoid overfitting and to generalize better.

![YOLO Training Process](/images/yolo-training-process.svg)
*The YOLO training process: from input image through CNN to predictions and weight updates.*

Well, now we have a trained YOLO model. What's next? Ready to put it to the test?

You can actually run real-time tests on this webpage: (http://pjreddie.com/yolo/)

Here is an example that I ran using YOLOv8 model that is already pre-trained and available in python.
You can find the complete code for this YOLO demonstration [in this GitHub Gist](https://gist.github.com/statisticalbits/b887dc9fac53ad7fab24cb379321ce28)


![YOLO OBJECT DETECTION](/images/Yolo_object_detection.png)
*The YOLO object detection: from real input image to object detection using YOLOv8 Model*

![YOLO PREDICTION VECTORS](/images/YOLO_Prediction_Vectors.png)
*The YOLO prediction vectors: Output of prediction vectors from YOLOv8 Model*


## Why is YOLO special?

Old methods looked at small patches of the images independently. YOLO divides the image into a grid, and each cell reasons globally (remember cell (1, 1) in our example predicts the dog's full box, even if the dog spills into other cells). This matters for context awareness as YOLO sees the whole scene, reducing false positives.

YOLO is a one-shot detection method that is simple, super fast, and efficient to deploy. BUT it trades slight accuracy for blazing speed.

Tesla's early FSD used YOLO-inspired models for camera-based detection, now replaced with more complex vision transformers.

Some autonomous vehicles use YOLO for fast initial detection but rely on LiDAR/sensor fusion for precision.

Drones like DJI use YOLO-like models for collision avoidance.

And there are many more applications of this cool research. Kudos to the researchers!

- [Original YOLO Paper](https://arxiv.org/abs/1506.02640)
- [YOLO Official Website](https://pjreddie.com/darknet/yolo/)
- [YOLOv4 Paper](https://arxiv.org/abs/2004.10934)
