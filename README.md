# Overview
A digit recognition system that uses GPU-accelerated logistic regression to classify handwritten digits from the MNIST dataset. 

# Features
- **GPU Acceleration**: Uses RAPIDS cuML for faster model training on NVIDIA GPUs
- **Accuracy**: Model achieved 92% accuracy score
- **Visualization**: Includes confusion matrix
- **Model Saving**: saving and loading the trained model using pickle

## Dataset
This project uses the **MNIST dataset** which contains:
- 70,000 handwritten digit images (28×28 pixels)
- 10 classes (digits 0-9)

# How It Works??
### 1- first the dataset is downloaded using fetch_openml:
```python
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.uint8)
```
### 2- split 20% of the dataset for testing and then normalize the pixel values:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train / 255.0
X_test = X_test / 255.0
```
### 3- model is trained on the data using cuLogisticRegression:
```python
model = cuLogisticRegression(
    solver='qn',
    max_iter=1000,
    tol=0.001,             # tolerance to stop when improvement is small
    output_type='numpy'    # return as numpy arrat instead of cuDF
)
model.fit(X_train, y_train)
```
### 4- open the a test image and apply thresholding to extract contours:
```python
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
### 5- after extracting each digit region its time to preprocess and predict:
```python
resize = cv2.resize(digit_img, (28, 28))
invert = 255 - resize
normalize = invert / 255.0
flatten = normalize.reshape(1, -1)

pred = model.predict(flatten)[0]
conf = model.predict_proba(flatten)[0]
```

## NOTE:
I've provided 2 models, one is trained using CPU the other is trained using GPU
You can test the GPU version using google colab

### Model Training Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lord-Mahdi1383/Digit_Recognition_With_Logistic_Regression/blob/main/MNIST_Logistic_Regression_GPU.ipynb)

### Model Testing Notebook On Custom Image:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lord-Mahdi1383/Digit_Recognition_With_Logistic_Regression/blob/main/Digit_Prediction.ipynb)

## FUTURE IMPROVEMENT:
- add a canvas to draw (using tkinter maybe) and a button to predict
- some digits have low prediction confidence (probably bad preprocessing)
