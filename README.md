A simple implementation for Human Detection

Steps performed:

1. Convolution operation using Sobel operator
2. Performing Histogram of Oriented gradient(HOG) operation on the images
3. Performing Local Binary Partion(LBP) measure on the images
4. Using a two layer Perceptron to train the feature vector obtained from step 3

We can notice that the accuracy of the Perceptron model increases when we take into account the HOG + LBP feature Vector compared to just HOG feature vector.

