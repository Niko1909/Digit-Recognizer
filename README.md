# Digit-Recognizer
A basic computer vision ML project on classifying the digit from a handwritten image, trained on the MNIST dataset, improving performance iteratively model by model. Python file is a readable version of the Jupyter Notebook file. The best model (model 3 or 4) achieves a validation accuracy of 99% and a test accuracy of 98.5%, slightly below the approach of finetuning a large pretrained model such as EfficientNet which can achieve a test accuracy of ~99.3%. Based on the Kaggle competition [here](https://www.kaggle.com/competitions/digit-recognizer/overview). 

## Libraries used:
* [Pandas](https://pandas.pydata.org/docs/index.html)
* [TensorFlow/Keras](https://www.tensorflow.org/)

## ML techniques used:
* Data preprocessing:
  * Standardization of pixel values
* Layers:
  * 2D Convolution
  * Batch Normalization
  * Dropout
  * Flatten
  * Dense
* Training:
  * Early stopping
  * Data augmentation: rescaling, random zoom, random rotation, random contrast
  * Memory optimization through tf.data.Dataset
  * Pretrained base model approach using VGG16
