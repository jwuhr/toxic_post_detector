# Toxic Comment Detector

This toxic comment detector aims to classify online comments based on their toxicity using deep learning. This model was built based on the training data provided by Kaggle for their "Toxic Comment Classification Challenge" 2 years ago.

The model uses long short-term memory (LSTM), and convolutional neural networks (CNN) to classify text into a list of toxic classifications. The current model achieves around a 98.2% accuracy using the training data set.

## Running the Program

### Step 1) Download Data from Kaggle

The training data, testing data, submission file, and GloVe vector file should all be downloaded and placed in the project's root directory.

You must download the training data from Kaggle, linked below:
- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

Below is the pre-trained english word vectors, provided by Stanford:
- [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

### Step 2) Create Directories

In the project directory, create a folder called "final" and another folder called "models". The "final folder will be the destination for the final submission CSV file, and the "models" folder will be for the trained model.

### Step 3) Train and Run Models

Once the training data and GloVe vector are downloaded in the project folder along with "toxic_comments.py", run the following commands on terminal in the project folder:

```
python toxic_comments.py
```

This will begin the process of training a new model, then will fill in the submission CSV file with the predicted toxicity values using the newly-created model.