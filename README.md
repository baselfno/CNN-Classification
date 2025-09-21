[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2axE5gl3)
# Cat-vs-Dog with CNN Classification Task

## Task Description
In this task, you will build a Convolutional Neural Network (CNN) for image classification. The goal is to classify images into two categories: **dogs** and **cats**. You will:

1. Preprocess and explore the dataset.
2. Build and train a CNN classification model.
3. Evaluate the model's performance using appropriate metrics.
4. Record your findings and observations under the **Findings** section.


## Dataset
The dataset used for this task is **Dogs vs. Cats**:

[Kaggle Dataset - Dog vs. Cat](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)

### Dataset Details:
- **Content**: Images of dogs and cats.
- **Format**: JPEG images.
- **Labels**: 0 for cat, 1 for dog.


## Requirements

1. **Model Requirements**:
   - Build a CNN model.
   - Include at least:
     - Input layer for image data.
     - Multiple convolutional layers with appropriate activation functions.
     - Pooling layers (e.g., MaxPooling).
     - Fully connected layers leading to a softmax or sigmoid output.
   - Use **binary cross-entropy** as the loss function for binary classification.

2. **Evaluation**:
   - Use metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
   - Create visualizations for:
     - Model training and validation loss.
     - Model training and validation accuracy.
     - Confusion matrix.

3. **Documentation**:
   - Clearly document:
     - The architecture of the CNN model.
     - The CNN (Convolutional Neural Network) model used here is designed for binary image classification to classify images into Cats and Dogs. The architecture consists of:

Layer Type	Details
Input Layer	Input shape: (128, 128, 3) (RGB images of size 128x128)
Conv2D Layer 1	Filters: 32, Kernel size: (3, 3), Activation: ReLU
MaxPooling2D Layer 1	Pool size: (2, 2)
Conv2D Layer 2	Filters: 64, Kernel size: (3, 3), Activation: ReLU
MaxPooling2D Layer 2	Pool size: (2, 2)
Dropout Layer 1	Dropout Rate: 30% (0.3)
Conv2D Layer 3	Filters: 128, Kernel size: (3, 3), Activation: ReLU
MaxPooling2D Layer 3	Pool size: (2, 2)
Flatten Layer	Flattens the feature map into 1D
Dense Layer 1	Units: 128, Activation: ReLU
Dropout Layer 2	Dropout Rate: 50% (0.5)
Dense Output Layer	Units: 1, Activation: Sigmoid
Activation Function:

ReLU (Rectified Linear Unit) for hidden layers.
Sigmoid activation for the final output layer since this is a binary classification problem.
     - Evaluation results.

## Findings
Document your results and observations here:
- **Accuracy**: [accuracy: 0.9882 ]
- **Loss**: [loss: 0.0430]
- **Observations**:
   - [E.g., CNN with 3 layers achieved better accuracy compared to 2 layers]
   - [Learning rate of 0.001 provided optimal convergence]

Add more details as needed to describe your experiments and outcomes.


## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Examples](https://keras.io/examples/)
- [Dataset on Kaggle](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)
