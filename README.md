# Assignment 8 — Handwritten Digit Recognition using ANN

## Objective
Automate the recognition of handwritten digits (0–9) — as would be
needed for reading postal codes — by building an Artificial Neural
Network (ANN) trained on the MNIST dataset.

## Dataset
- MNIST Handwritten Digits Dataset (Kaggle, CSV format):
  https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
- The dataset is **not included** in this repository. Download
  `mnist_train.csv` and `mnist_test.csv` from the link above and place
  them in this folder before running the script.
- If the CSVs are not found, the script automatically falls back to
  `tensorflow.keras.datasets.mnist` (fetched by Keras itself) so the
  pipeline can still be run end-to-end.

## Libraries Used
- `pandas`, `numpy` — data handling
- `tensorflow` / `keras` — building, compiling, and training the ANN
- `scikit-learn` — train/test split, confusion matrix, classification
  report
- `matplotlib`, `seaborn` — sample digit display, confusion matrix, and
  accuracy/loss curves

## Methodology
1. **Data Understanding** — Load the dataset, display the first five
   records, identify input features (784 pixel values per image) and the
   target variable (digit label 0–9), report dataset dimensions, and
   display one sample digit image.
2. **Preprocessing** — Check for missing values, separate features and
   target, normalize pixel values to [0, 1], split 80/20 into
   train/validation sets, and one-hot encode the labels.
3. **Model Development** — Train the ANN described below for 10 epochs
   and generate predictions on the test set.
4. **Evaluation** — Compute test accuracy, confusion matrix,
   classification report, and plot accuracy/loss vs. epoch curves.

## Model Architecture
| Layer            | Details                  |
|-------------------|---------------------------|
| Input             | 784 (flattened 28×28 image) |
| Hidden Layer 1    | 128 neurons, ReLU          |
| Hidden Layer 2    | 64 neurons, ReLU           |
| Output Layer      | 10 neurons, Softmax        |

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Metric:** Accuracy
- **Epochs:** 10

## Results
Running `Assignment-8.py` saves:
- `sample_digit.png` — an example digit from the dataset
- `confusion_matrix.png` — confusion matrix heatmap
- `accuracy_vs_epoch.png`, `loss_vs_epoch.png` — training curves
- `ann_mnist_model.keras` — the trained model

Test accuracy, the full confusion matrix, and classification report are
printed to the console, along with 4 model-performance observations.

## Conclusion
See the `Conclusion` section printed by `Assignment-8.py`, covering key
findings, the importance of hidden layers, one advantage of deep learning
over traditional ML, and one limitation of ANNs for image tasks.

## How to Run
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
# Place mnist_train.csv and mnist_test.csv (from Kaggle) in this folder, then:
python Assignment-8.py
```

## Author
**Sajjad Shaik**
GitHub: [SajjadShaik2005](https://github.com/SajjadShaik2005)
Email: sajjad102005@gmail.com
