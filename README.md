\# CIFAR-10 Image Classifier (PyTorch)



A Convolutional Neural Network (CNN) built from scratch in PyTorch to classify 32x32 pixel images from the CIFAR-10 dataset into 10 distinct categories. 



\*\*Note:\*\* This project was developed as a foundational machine learning exercise to practice building, training, and evaluating custom computer vision architectures under strict hardware constraints.



\## Hardware \& Optimization

This model was specifically engineered to train locally on an entry-level \*\*NVIDIA GeForce MX330 GPU\*\* with only \*\*2GB of VRAM\*\*. 

To prevent Out of Memory (OOM) errors while maximizing accuracy, the pipeline utilizes:

\* Strict mini-batching (batch size of 64).

\* In-place gradient zeroing and memory caching clearance.

\* Efficient `torch.no\_grad()` evaluation blocks.



\## Features \& Architecture

The final model (`AdvancedCNN`) achieved \*\*\~85% accuracy\*\* on the 10,000-image test set. 

Key techniques implemented include:

\* \*\*Data Augmentation:\*\* Random cropping and horizontal flipping to artificially expand the training distribution.

\* \*\*Batch Normalization:\*\* Applied between convolutional layers to stabilize mathematical outputs and allow for a deeper network.

\* \*\*Dropout Regularization:\*\* Progressively dropping 20% to 50% of neurons during training to prevent co-adaptation and overfitting.

\* \*\*Learning Rate Scheduling:\*\* Implementing `StepLR` to decay the learning rate and fine-tune weights in later epochs.



\## Evaluation

The model was evaluated using Scikit-Learn and Seaborn to generate a Confusion Matrix and classification report. 

\* Highest performing classes: `car`, `truck`, `ship` (F1-Scores > 0.90)

\* Lowest performing classes: `cat`, `bird` (F1-Scores < 0.75)

\* \*Observation:\* The network excels at identifying mechanical/geometric shapes with predictable backgrounds, but struggles with the soft edges and varied poses of animals (frequently confusing cats and dogs).



\## How to Run



1\. Clone the repository and install requirements (`torch`, `torchvision`, `matplotlib`, `seaborn`, `scikit-learn`).

2\. Run `python train\_advanced.py` to download the CIFAR-10 dataset and train the model. (The `.pth` weights and `data` folder are excluded via `.gitignore`).

3\. Run `python evaluate.py` to test the model against the 10,000 unseen test images and generate the confusion matrix heatmap.

4\. Run `python predict.py` to pull 5 random images and view the model's live confidence predictions.

