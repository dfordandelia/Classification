# Course Project: EE 769: Introduction to Machine Learning

# For the wine dataset (both red and white)
This code performs several tasks related to analyzing and modeling white-wine data:

1. # Data Loading and Inspection:
   - It reads white-wine data from a CSV file and examines for missing values.

2. # Data Preprocessing:
   - It separates the features (independent variables) from the target variable ('quality').

3. # Data Visualization:
   - It visualizes the distribution of each feature using histograms.
   - It visualizes the correlation between features using a heatmap.

4. # Model Training and Hyperparameter Tuning:
   - It splits the dataset into training and validation sets.
   - It iterates through different combinations of hyperparameters for Random Forest Classifier (`n_estimators` and `max_depth`) and Support Vector Regressor (`C` and `epsilon`).
   - It evaluates each combination's performance using accuracy score for Random Forest Classifier and mean squared error for Support Vector Regressor on the validation set.
   - It selects the best model based on the highest accuracy for Random Forest Classifier and the lowest mean squared error for Support Vector Regressor.

5. # Feature Importance:
   - It calculates and visualizes the feature importance for the Random Forest Classifier.

6. # Model Evaluation:
   - It prints the best performing Random Forest Classifier model along with its validation accuracy.
   - It prints the best performing Support Vector Regressor model along with its validation mean squared error.

# For the mice data
This code performs several tasks for analyzing and modeling mice data:

1. # Data Loading and Inspection:
   - It reads the mice data from an Excel file and displays the data.
   - Calculates the number of missing values in each column.

2. # Data Preprocessing:
   - Removes columns with more than three missing values.
   - Selects the features columns and drops unnecessary columns like 'MouseID'.
   - Visualizes the correlation among features using a heatmap and drops features with high correlation values.

3. # Data Imputation:
   - Imputes missing data using IterativeImputer.

4. # Feature Engineering:
   - Prepares the data for modeling by separating features and target vectors.
   - Converts categorical data into binary form.

5. # Model Training and Hyperparameter Tuning (Random Forest Classifier):
   - Splits the dataset into training and validation sets.
   - Iterates through different combinations of hyperparameters for Random Forest Classifier (`n_estimators` and `max_depth`) and evaluates the model's performance using accuracy score on the validation set.
   - Selects the best performing Random Forest Classifier model.

6. # Model Training and Hyperparameter Tuning (Neural Network):
   - Builds and trains a neural network with different numbers of hidden neurons.
   - Evaluates the model's performance using accuracy score on the validation set.
   - Selects the best performing neural network model.

7. # Feature Selection (Recursive Feature Elimination with Cross-Validation):
   - Uses RFECV to select the optimal number of features for the Random Forest Classifier.
   - Trains the model on the selected features and evaluates its performance.

8. # Model Training and Hyperparameter Tuning (Support Vector Regressor):
   - Iterates through different combinations of hyperparameters for Support Vector Regressor (`C` and `epsilon`) and evaluates the model's performance using mean squared error on the validation set.
   - Selects the best performing Support Vector Regressor model.

# For the hymenoptera data (link to the dataset: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
The provided code is structured to perform a comprehensive analysis of different machine learning models for a binary classification task using a dataset of images. Here's a detailed description of each section:

1. # Data Preprocessing and Model Evaluation Setup:
   - The initial sections of the code set up data transformations (`data_transforms`) for image preprocessing and normalization.
   - It also prepares the dataset by loading images from specified directories (`image_datasets`) for training and validation, creating data loaders (`dataloaders`) for efficient batch processing, and calculating dataset sizes (`dataset_sizes`) and class names (`class_names`).
   - Additionally, it sets up the computing device (`device`) to utilize GPU if available.

2. # Visualization Functions:
   - Two functions (`imshow` and `visualize_model`) are defined to visualize images and model predictions.
   - These functions are utilized to visualize predictions made by the model on a batch of validation images.

3. # Model Training and Evaluation Functions:
   - Two functions (`train_model` and `visualize_model_predictions`) are defined to train a model and visualize its predictions, respectively.
   - The `train_model` function performs model training using a specified criterion, optimizer, and scheduler. It also evaluates the model's performance on a validation set and prints the best validation accuracy achieved during training.
   - The `visualize_model_predictions` function takes a trained model and an image path as input, preprocesses the image, makes predictions using the model, and visualizes the predicted class along with the image.

4. # Fine-tuning ResNet-18 Model:
   - The code initializes a ResNet-18 model pretrained on ImageNet (`models.resnet18(weights='IMAGENET1K_V1')`).
   - It modifies the final fully connected layer of the model for binary classification (`nn.Linear(num_ftrs, 2)`).
   - Cross-entropy loss (`nn.CrossEntropyLoss()`) is chosen as the loss function.
   - Stochastic Gradient Descent (SGD) optimizer (`optim.SGD`) is utilized for parameter optimization, with a learning rate scheduler (`lr_scheduler.StepLR`) to decay learning rate by a factor of 0.1 every 7 epochs.

5. # Training and Evaluation of Fine-tuned ResNet-18 Model:
   - The fine-tuned ResNet-18 model is trained and evaluated using the `train_model` function, which performs training and validation loops and prints the best validation accuracy achieved.
   - The performance of the model on the validation set is visualized using the `visualize_model_predictions` function.

Overall, this code encapsulates a complete pipeline for training, evaluating, and visualizing predictions made by machine learning models, with a focus on fine-tuning a pretrained ResNet-18 model for a binary image classification task.
