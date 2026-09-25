# Fashion-MNIST Rotation-Robust Classification

This project was completed as part of my machine learning coursework at UC Berkeley.

The goal of this project was not only to classify Fashion-MNIST images, but also to study how image transformations affect model performance and how to improve robustness when test data differs from training data.

The original classifier was trained mostly on upright Fashion-MNIST images, while a later test set contained rotated images. This created a distribution shift and motivated several strategies for improving classification performance.

---

## Project Overview

This project covers a full machine learning workflow, including:

- Image preprocessing and standardization
- K-means clustering and visualization
- MLP classification
- Class-wise accuracy analysis
- Confusion matrices
- Prediction confidence analysis
- Image augmentation
- Geometric image transformations
- Bilinear interpolation
- Rotation-angle regression
- Distribution shift
- Rotation correction
- Test-time augmentation

---

## Image Transformations

I implemented several image transformations and used them to study how changes in image orientation and appearance affect classifier performance.

### Horizontal Flip

Implemented horizontal image flipping to generate mirrored versions of Fashion-MNIST images.

### Image Shifting

Implemented horizontal and vertical image translation while preserving the original image dimensions.

### Image Blurring

Implemented image blurring to study how reduced visual detail affects classification performance.

---

## Image Rotation with Linear Algebra

Instead of relying only on a high-level image-processing library, I implemented image rotation using linear algebra.

For each pixel, I first translated the coordinate system so that the center of the image became the origin.

Then I applied the standard 2D rotation matrix:

\[
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
\]

After rotation, I translated the coordinates back into the image coordinate system.

Each Fashion-MNIST image is 28 × 28 pixels, so it can also be represented as a 784-dimensional vector.

I constructed a transformation matrix

\[
T \in \mathbb{R}^{784 \times 784}
\]

so that a flattened image could be transformed as:

\[
x_{\text{rotated}} = T x
\]

This connected image processing directly to linear algebra concepts such as:

- Rotation matrices
- Coordinate translation
- Linear transformations
- Matrix-vector multiplication
- Flattened image representations
- Pixel index mapping

![Examples of image rotation at different angles](images/rotation-examples.png)

*Original Fashion-MNIST image compared with rotations at 45°, 90°, 200°, and 270°.*

---

## Bilinear Interpolation

A basic rotation can create gaps and visual artifacts because transformed coordinates do not always land exactly on integer pixel locations.

To improve the rotated images, I implemented bilinear interpolation.

Instead of assigning an output pixel from only one nearest input pixel, bilinear interpolation estimates the new pixel value using weighted contributions from neighboring pixels.

This produces smoother rotated images and reduces gaps introduced during geometric transformation.

![Rotation compared with bilinear interpolation](images/bilinear-interpolation-comparison.png)

*Comparison of the original image, basic 45° rotation, and 45° rotation using bilinear interpolation.*

---

## Transformation Composition and Data Augmentation

I also combined several transformations to create more diverse training examples.

Transformations included:

- Horizontal and vertical shifting
- Rotation
- Bilinear rotation
- Blur
- Rotation followed by blur
- Multiple composed transformations

This helped me understand that the order of transformations matters and that multiple image operations can be composed into a more complex augmentation pipeline.

![Examples of image augmentation](images/image-augmentation-examples.png)

*Examples of shifting, bilinear rotation, blurring, and composed image transformations used for augmentation.*

---

## Classification Model

I trained a multilayer perceptron classifier on Fashion-MNIST using scikit-learn.

The classification pipeline included:

- Train/test splitting
- Feature standardization using `StandardScaler`
- MLP classifier training
- Prediction
- Overall classification accuracy
- Class-wise accuracy analysis
- Confusion matrix analysis
- Prediction-confidence analysis

I also examined which Fashion-MNIST classes were easiest and hardest for the model to distinguish.

---

## Handling Rotated Test Images

A major challenge appeared when evaluating the classifier on a new test set.

The training images were mostly upright, while the new test set contained rotated images.

This created a distribution shift:

> The model was trained on one image distribution but evaluated on another.

I explored three different approaches to address this problem.

---

## Solution 1: Rotation-Augmented Training

The first approach was to make the training data more diverse.

I generated rotated versions of training images and added them to the training set before retraining the classifier.

### Simplified Idea

```python
for image in training_images:
    angle = sample_rotation()

    rotated_image = rotate(
        image,
        angle
    )

    augmented_training_set.append(
        rotated_image
    )
```

The goal was to expose the classifier to different image orientations during training so that it would become less sensitive to rotation.

---

## Solution 2: Predict and Undo the Rotation

The second approach treated image rotation as a separate regression problem.

I trained a regression model to predict the rotation angle of an image.

For each rotated test image:

1. Predict its rotation angle
2. Rotate the image in the opposite direction
3. Standardize the corrected image
4. Classify the corrected image

### Simplified Pipeline

```python
predicted_angle = rotation_model.predict(image)

corrected_image = rotate(
    image,
    -predicted_angle
)

prediction = classifier.predict(
    corrected_image
)
```

![Rotation prediction and correction](images/rotation-prediction-correction.png)

*Example of the rotation-correction pipeline: original image, rotated image, and the image after applying the model's predicted correction.*

This approach attempts to transform the test image back toward the distribution seen during training.

---

## Rotation Regression

I trained a regression model to estimate image rotation angles.

The rotation model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R²
- Predicted vs. true rotation-angle comparisons

The regression model became an important part of both rotation correction and test-time augmentation.

---

## Solution 3: Test-Time Augmentation

The third approach was the most interesting part of the project for me.

Instead of trusting a single predicted rotation angle, I searched across multiple nearby correction angles and allowed the classifier to evaluate several candidate orientations.

The process was:

1. Predict an initial rotation angle
2. Perform a coarse search across rotation offsets
3. Measure the residual rotation of each corrected image
4. Identify the most promising candidate regions
5. Perform a finer local search around those regions
6. Classify all corrected versions
7. Use classifier confidence to choose the final prediction

This created a two-stage search strategy:

> coarse search → local refinement → confidence-based prediction

### Simplified Algorithm

```python
predicted_angle = rotation_model.predict(image)

candidate_offsets = generate_coarse_offsets()

candidates = []

for offset in candidate_offsets:

    corrected = rotate(
        image,
        -predicted_angle + offset
    )

    residual_angle = rotation_model.predict(
        corrected
    )

    candidates.append(
        (offset, abs(residual_angle))
    )

best_regions = select_low_residual_candidates(
    candidates
)

local_offsets = fine_search(
    best_regions
)

corrected_images = [
    rotate(
        image,
        -predicted_angle + offset
    )
    for offset in local_offsets
]

probabilities = classifier.predict_proba(
    corrected_images
)

prediction = choose_highest_confidence_class(
    probabilities
)
```

This approach combines two different signals:

- The rotation-regression model estimates how close the image is to an upright orientation
- The classifier provides confidence about the predicted clothing class

Instead of relying on one correction angle, the method searches several plausible orientations and uses model confidence to select the final prediction.

---

## Model Evaluation

Throughout the project, I used several evaluation methods:

- Classification accuracy
- Class-wise accuracy
- Confusion matrices
- Prediction confidence
- MAE
- MSE
- RMSE
- R²
- Predicted vs. true rotation-angle analysis

I also compared different strategies for handling rotated test data.

---

## Code Highlights

The following examples summarize some of the main ideas I implemented.

```python
# Rotation using a transformation matrix

theta = np.deg2rad(theta)

for i_src in range(height):
    for j_src in range(width):

        # Move image center to the origin
        x = j_src - center_j
        y = i_src - center_i

        # Apply 2D rotation
        x_rot = (
            np.cos(theta) * x
            - np.sin(theta) * y
        )

        y_rot = (
            np.sin(theta) * x
            + np.cos(theta) * y
        )

        # Translate coordinates back
        j_out = round(x_rot + center_j)
        i_out = round(y_rot + center_i)

        if inside_image(i_out, j_out):
            T[output_index, input_index] = 1


# Rotation correction

predicted_angle = rotation_model.predict(image)

corrected_image = rotate(
    image,
    -predicted_angle
)

prediction = classifier.predict(
    corrected_image
)


# Test-time augmentation

predicted_angle = rotation_model.predict(image)

coarse_candidates = search_rotation_offsets(
    image,
    predicted_angle
)

best_regions = select_best_candidates(
    coarse_candidates
)

local_candidates = refine_search(
    best_regions
)

probabilities = classifier.predict_proba(
    local_candidates
)

prediction = choose_highest_confidence_class(
    probabilities
)
```

---

## Technologies & Concepts

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Plotly
- Google Colab
- Fashion-MNIST
- K-means Clustering
- MLP Classification
- Regression
- Standardization
- Image Augmentation
- Transformation Matrices
- Rotation Matrices
- Linear Transformations
- Matrix-Vector Multiplication
- Bilinear Interpolation
- Distribution Shift
- Rotation Correction
- Test-Time Augmentation
- Confidence-Based Prediction
- Model Evaluation

---

## What I Learned

This project helped me understand that model performance depends not only on the model itself, but also on how closely the test data matches the training distribution.

A classifier that performs well on upright Fashion-MNIST images can perform much worse when the same types of images are rotated.

I especially enjoyed comparing three different approaches to this problem:

1. Making the training data more diverse
2. Correcting test images before classification
3. Searching over multiple transformations during inference

The image-rotation section also helped me connect linear algebra directly to image processing. Instead of thinking of rotation as only a library function, I worked with coordinate systems, rotation matrices, matrix-vector multiplication, and interpolation.

The test-time augmentation approach was especially interesting because it showed that improving predictions does not always require retraining the classifier. The inference process itself can also be made more robust.

---

## Course Context

This project was completed as part of my machine learning coursework at UC Berkeley.

The course framework, dataset setup, and assignment structure were provided as part of the course. This repository focuses on documenting the techniques I implemented, the experiments I performed, and the ideas I explored rather than publishing the original course solution.
