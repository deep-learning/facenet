# Face Recognition using Tensorflow [![Build Status][travis-image]][travis]

[travis-image]: http://travis-ci.org/davidsandberg/facenet.svg?branch=master
[travis]: http://travis-ci.org/davidsandberg/facenet

## Possible Improvements

- Add random translation and horizontal flipping of images? (already done)
  - This has been added some time ago (facenet_train.py --random_crop --random_flip ...). However it does not give any imporvement on the results.
- train with better face alignment (https://github.com/davidsandberg/facenet/issues/19)
  - MTCNN has been shown to improve the performance significantly, so that should be used instead.
- How to improve accuracy 
  - Yes, I guess this is the main question and I don't have the final answer. But I think part of the answer will be to use a larger set of training data, for example a combination of the VGG Face dataset and FaceScrub. But it will most likely include a larger network (including larger input images).
   In addition to this there will most certainly be other things that help to improve performance, like using dropout, use augumented training data (flipping and translations), maybe better weight initialization etc etc.
 - But if you want to do image classification this is not as straight forward. The best way is to replace the top layer of the model (i.e. the linear mapping to the 128-dimensional embedding) with a linear mapping to a vector with the same dimension as there are classes, and then calculate a loss using soft max.
  I have recently started to work on this, mainly in order to get better filters, but it is far(!) from finished. But you can have a look at it on the branch classifier_pretrain. The facenet_train module can be found here.
  The code seems to be running and learning something but that's all i can say...
- training dataset and testing dataset should use same face alignment so LFW dataset and training dataset should use same face alignment and I reproduce the same accurary as the author did
- MS-Celeb-V1 (https://github.com/davidsandberg/facenet/issues/48)
- https://github.com/davidsandberg/facenet/issues/48
- https://github.com/davidsandberg/facenet/issues/152
- train step (https://github.com/davidsandberg/facenet/issues/49)
- how much time does it take you to train a model with 0.919 accuracy on LFW?
  -  It depends on a few factors, e.g. how fast your disk is, how fast the triplet selection can run on the CPU and how fast the training on the GPU is. For me it takes roughly 72 hours to train for 500 000 steps with triplet loss and the nn4 model. 
- Yes, the embeddings that are calculated by facenet are in the Euclidean space where distances
  directly correspond to a measure of face similarity. The embeddings can then be used for verification, recognition and clustering.
- The MTCNN is very good at detecting profile faces which is very nice. But it's not clear to me how to apply a 2D transformation that does not cause severe distortions to e.g. profile faces (where estimated positions for the eyes will be in the same position). I know that for example DeepFace uses 3D alignment which seems to work pretty well, but I guess it becomes algorithmically more tricky.
  So far my approach has been to just use the face bounding box and let the model generalize over different face poses.
- I understand. Well, in my opinion a 2D alignment wouldn't distort the image, as we just need to rotate in a way that the face stays in vertical position (just think of cases where the neck is bent to one side and the face is not completely in vertical position). The reason is that the convolution is invariant to translation but not rotation, so I think it would improve in some way. I am going to perform some experiments with 2D alignment, I also have code for 3D alignment that I will try as well.
  - @davidsandberg, I tried the 2D alignment as I mentioned above, then I got this:
    
    Runnning forward pass on LFW images
    Accuracy: 0.985+-0.006
    Validation rate: 0.90600+-0.02119 @ FAR=0.00069
    
    Any thoughts? It is a very slightly improvement I guess
- Training and validation image pre-processing
  - Hi @davidsandberg, can you explain what pre-processing step do you apply during the training and the validation process. I notice that in the training code (classification) the images are not pre-whitened and the features are not l2 normalized. This is not the case in the validation step. Is it right ?
```aidl
For training using facenet_train_classifier.py the preprocessing is done in facenet.read_and_augument_data(...) as

 for _ in range(nrof_preprocess_threads):
        image, label = read_images_from_disk(input_queue)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        #pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_whitening(image)
        images_and_labels.append([image, label])
So there is a pre-whitening step here as well.
```
- what is the image size for those pre-trained model
  - Since it's a CNN the input size is not that crucial and it can work fine with other image sizes as well. But of course, if the input images are too small the number of activations in the higher layers of the network will become to few and then it will crash. Also, if the input image size is changed the number of parameters in the resulting CNN will change as well, which means that training could require a different weight decay etc.
- facenet pruning
- 



![Travis](http://travis-ci.org/davidsandberg/facenet.svg?branch=master)

This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf) as well as the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Compatibility
The code is tested using Tensorflow r1.2 under Ubuntu 14.04 with Python 2.7 and Python 3.5. The test cases can be found [here](https://github.com/davidsandberg/facenet/tree/master/test) and the results can be found [here](http://travis-ci.org/davidsandberg/facenet).

## News
| Date     | Update |
|----------|--------|
| 2017-05-13 | Removed a bunch of older non-slim models. Moved the last bottleneck layer into the respective models. Corrected normalization of Center Loss. |
| 2017-05-06 | Added code to [train a classifier on your own images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images). Renamed facenet_train.py to train_tripletloss.py and facenet_train_classifier.py to train_softmax.py. |
| 2017-03-02 | Added pretrained models that generate 128-dimensional embeddings.|
| 2017-02-22 | Updated to Tensorflow r1.0. Added Continuous Integration using Travis-CI.|
| 2017-02-03 | Added models where only trainable variables has been stored in the checkpoint. These are therefore significantly smaller. |
| 2017-01-27 | Added a model trained on a subset of the MS-Celeb-1M dataset. The LFW accuracy of this model is around 0.994. |
| 2017&#8209;01&#8209;02 | Updated to code to run with Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.   |

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE) | 0.987        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) | 0.992        | MS-Celeb-1M      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## Inspiration
The code is heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on a subset of the [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) dataset. This dataset is significantly larger but also contains significantly more label noise, and therefore it is crucial to apply dataset filtering on this dataset.

## Pre-processing

### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set to "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.

## Running training
Currently, the best results are achieved by training the model as a classifier with the addition of [Center loss](http://ydwen.github.io/papers/WenECCV16.pdf). Details on how to train a model as a classifier can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1).

## Pre-trained model
### Inception-ResNet-v1 model
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model. The datasets has been aligned using [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Performance
The accuracy on LFW for the model [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) is 0.992+-0.003. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw).
