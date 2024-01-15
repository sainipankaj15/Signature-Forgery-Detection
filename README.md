# Signature Forgery Detection Documentation

## Overview

The provided code implements a signature forgery detection system using a neural network. The system processes signature images, extracts various features from them, and then trains a neural network to distinguish between genuine and forged signatures. The features include ratio, centroid, eccentricity, solidity, skewness, and kurtosis.

## Code Structure

### Libraries Used
- `numpy`: For numerical operations.
- `os`: For interacting with the operating system (file paths, folder creation).
- `matplotlib`: For image visualization.
- `scipy.ndimage`: For image processing operations.
- `skimage.measure.regionprops`: For extracting region properties from binary images.
- `skimage.io`: For reading images.
- `skimage.filters.threshold_otsu`: For thresholding images.
- `tensorflow`: For creating and training neural networks.
- `pandas`: For handling CSV files.

### Functions

1. **`rgbgrey(img)`**
   - Converts RGB images to grayscale.
   
2. **`greybin(img)`**
   - Converts grayscale images to binary using Otsu's thresholding.
   
3. **`preproc(path, img=None, display=True)`**
   - Preprocesses the signature image by converting it to grayscale, binary, and extracting the signature part using a bounding box.
   
4. **`Ratio(img)`**
   - Computes the ratio of white pixels to the total number of pixels in the binary image.
   
5. **`Centroid(img)`**
   - Computes the centroid (center of mass) of the white pixels in the binary image.
   
6. **`EccentricitySolidity(img)`**
   - Computes the eccentricity and solidity of the signature using region properties.
   
7. **`SkewKurtosis(img)`**
   - Computes the skewness and kurtosis of the signature.
   
8. **`getFeatures(path, img=None, display=False)`**
   - Extracts various features from the signature image using the above functions.
   
9. **`getCSVFeatures(path, img=None, display=False)`**
   - Extracts features and formats them for saving to a CSV file.
   
10. **`makeCSV()`**
    - Creates CSV files for training and testing data, containing features and corresponding labels (genuine or forged).
    
11. **`testing(path)`**
    - Creates a CSV file for a test image containing its features.
    
12. **`readCSV(train_path, test_path, type2=False)`**
    - Reads data from CSV files, returning training and testing data.
    
13. **`multilayer_perceptron(x)`**
    - Defines the structure of the neural network model.
    
14. **`evaluate(train_path, test_path, type2=False)`**
    - Trains the neural network and evaluates its performance on training and testing data.
    
15. **`trainAndTest(rate=0.001, epochs=1700, neurons=7, display=False)`**
    - Trains and tests the model with specified hyperparameters and returns average accuracy.

## Workflow

1. **Image Preprocessing:**
   - Convert RGB image to grayscale.
   - Convert grayscale image to binary using Otsu's thresholding.
   - Extract the signature part using a bounding box.

2. **Feature Extraction:**
   - Extract various features such as ratio, centroid, eccentricity, solidity, skewness, and kurtosis from the preprocessed image.

3. **Data Preparation:**
   - Create CSV files for training and testing data, including features and labels.

4. **Neural Network Training:**
   - Train a neural network using the extracted features as input and labels as output.

5. **Evaluation:**
   - Evaluate the trained model on both training and testing datasets.

## Usage

1. **Training Data Preparation:**
   - Images of genuine and forged signatures are stored in the "real" and "forged" folders, respectively.
   - Running `makeCSV()` will create CSV files with features and labels for training and testing.

2. **Model Training and Evaluation:**
   - Run `trainAndTest()` to train and evaluate the model with specified hyperparameters.
   - The function outputs average training accuracy, testing accuracy, and time taken for evaluation.

3. **Testing a Single Image:**
   - Run `testing(path)` to create a CSV file with features for a single test image.

4. **Customization:**
   - The neural network architecture and hyperparameters can be adjusted in the code for experimentation.

5. **Note:**
   - The code uses TensorFlow v1, as indicated by `tf.compat.v1`. Consider updating to the latest TensorFlow version if necessary.

## Example Usage

```python
# Set genuine and forged image paths
genuine_image_paths = "real"
forged_image_paths = "forged"

# Create CSV files for training and testing
makeCSV()

# Provide training person's id and test image path
train_person_id = input("Enter person's id : ")
test_image_path = input("Enter path of signature image : ")

# Set paths for training and testing CSV files
train_path = 'Features/Training/training_' + train_person_id + '.csv'
testing(test_image_path)
test_path = 'TestFeatures/testcsv.csv'

# Evaluate the trained model
train_accuracy, test_accuracy, evaluation_time = trainAndTest()

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Time taken for evaluation:", evaluation_time)
```

## Conclusion

This documentation provides an overview of the signature forgery detection system, including its code structure, functions, workflow, and example usage. Users can utilize this system for signature forgery detection with customization options for neural network hyperparameters.
