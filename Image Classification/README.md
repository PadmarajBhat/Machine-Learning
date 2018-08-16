# Image Classification on  CIFAR-10 Dataset

Flow of the python 3.6 program:
1. Import the Data : Download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Explore Data : explore the dataset batches, browse through the lables and respective images.
3. Preprocessing : Normalize the images, One hot encoding for the lables and finally, save the preprossed data.
4. Build a Convolutional Model: Experiment with various combination of convolution, max pool, drop out and fully connect layers to have a better Validation Accuracy. When satisfactory accuracy is reached, save the model for testing and for future use.
5. Test model
