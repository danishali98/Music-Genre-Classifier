# Music Genre Classifier

The repository contains the code to train a Music Genre Classifier using:
  1. Convolutional Neural Network (CNN)
  2. kNN (k Nearest Neighbors)

The dataset for the models in *GTZAN* dataset which contains audio clips from a wide set of genres.

The file *music_genre.py* contains a self implementation of the kNN model and it implements its own logic. It also provides you with the test accuracy of the model using Train/Test split.

The *genre_test.py* can be used to test a new audio clip for genres considering that the clip be in .wav format.

The file *music_genre_cnn.py* contains the implementation of CNN model from skLearn. It first creates a spectogram form the audio clips. Then, features are extracted from the spectogram and written to a *.csv* file. For that you have to uncomment the respective code as highlighted by comments in the code. Then for tuning your model, you can comment out the spectogram creation and feature extraction.

The model consists of 4 layers of with the Rectified Linear Unit (relu) with Regularizers and a softmax layer with 10 neurons indicating the total of 10 supported genres.

# How to Run?
Download the GTZAN music dataset available online and change the dataset loading path in respective files. Simple run the models and get test accuracies!
