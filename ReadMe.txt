STEPS:

1. First extract the positive & negative images of size 16x32 (extractImages.py)
2. Then find the dct transform of the images and take only first 21 values of the dct transform in forming the descriptor.
3. Set the block-size as 8x8, block stride as 4, so the number of function call becomes 21, and thus the size of the descriptor is 441, which serves as the features of a sample/image
4. Then train 1000 weak classifiers by randomly selecting a few training samples with randomly selected features on naive bayes classifier
   for best accuracy, we've used: no. of samples = 100, no. of features = 300 & thereshold error should be less than 0.06.
   Best accuracy achieved is 96.7%

NOTE:

1. CC-ICA(Fast-ICA) is to be implemented prior to the classification.
   