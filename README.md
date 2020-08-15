# Siamese Neural Network for One-shot Image Recognition
Machine Learning Model using Convolutional Architecture to naturally rank similarity between given inputs in order to make correct predictions given only single example of a new character image class.

# Benefits:
1. Model can make predictions given only single image of a new class.
2. Prediction classes can be added/removed without need for retraining the model.

## Dataset
Used Omniglot dataset which is a collection of 1623 hand drawn characters from 50 different alphabets. For every character there are just 20 examples, each drawn by a different person. Each image is a gray scale image of resolution 105x105.

## Implementation
Implementation is broadly divided into 4 steps:
1. Loading dataset
2. Mapping the problem to binary classification task
3. Model Architecture and Training 
4. Testing
# 1. Loading dataset
The datafiles have this format :

main_file -> alphabets(different languages) -> letters(for particular alphabet) -> 20 images for each letter

function loadimgs groups all letter images in X, marks corresponding labels in y, and maps the ranges of letter's labels in an alphabet in lang_dict.


Sample example:

Let X.shape is (964,20,105,105),this means we have 964 characters (or letters or categories) spanning across 30 different alphabets. For each of this character, we have 20 images, and each image is a gray scale image of resolution 105x105. 

Also Y.shape is (19280,1) signifies Total number of images = 964 * 20 = 19280. All the images for one letter have the same label.

Similarly for lang_dict['Sanskrit'] = [110, 136], denotes letters of Sanskrit alphabet have labels ranging from 110 to 136 (inclusive).

# 2. Mapping the problem to binary classification task

We can map this problem into a supervised learning task where our dataset contains pairs of (Xi, Yi) where ‘Xi’ is the input and ‘Yi’ is the output.

Xi contains the pair of images

Yi = 1 when both images belongs to same letter, else 0.

# 3. Model Architecture and Training 

Two CNN architecture each having 4 conv layers followed by maxpool layers except the last one, which after flattening is followed by fully connected layer of size 4096.
Both networks are integrated by taking difference of outputs for fully connected layer, which is fed to final output layer with sigmoid activation.

Pictorial representation of network:
![image](https://miro.medium.com/max/4268/1*v40QXakPBOmiq4lCKbPu8w.png)



# 4. Te



# References - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
