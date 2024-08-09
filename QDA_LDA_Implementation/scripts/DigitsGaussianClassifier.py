import sys

import scipy.cluster.vq

if sys.version_info[0] < 3:
	raise Exception("Python 3 not detected.")
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
import numpy as np
import matplotlib.pyplot as plt
import gaussian
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from save_csv import results_to_csv


def contrast_normalize(vector):
    norm = np.linalg.norm(vector)
    normalized = vector / norm if norm > 0 else vector
    return normalized


mnist_data = np.load('../data/mnist-data-hw3.npz')

data_length = mnist_data['training_data'].shape[0]
reshaped_data = mnist_data['training_data'].reshape(data_length, -1)

num_features = reshaped_data.shape[1]


"""
 Whitening data works very poorly when used with QDA, 
 understandably as it gets rid of the correlation between our variables. With LDA, whitening the data had better results
 I need to look for some feature engineering that works well with QDA, Contrast-normalization was mentioned.
 I tired normalizing by dividing pixels by their l2-norm, improvement was negligible.
 I want to experiment with HOG features using QDA.
 The error rate might also be affected by the regularization parameter used for singular matrices, there are many ways to deal
 with PSD covariance matrices, I'm using the simpleset way, adding alpha to the diagonal. 
 sklearn uses a somewhat similar trick. scipy.multivariate_normal.logpdf uses shrinkage (PCA).
 
 
 
 LDA and whitening paper for future reference: 
 
 WDiscOOD: Out-of-Distribution Detection via Whitened Linear Discriminant Analysis
 whitening uses the inverse square root of the covariance matrix (maps ellipsoids to spheres)
"""

normalised_mnist_data = scipy.cluster.vq.whiten(reshaped_data)



data_train, data_test, label_train, label_test = train_test_split(
    normalised_mnist_data, mnist_data['training_labels'], test_size=0.16666, random_state=42)


# def plot_covariance():
#     plt.figure(figsize=(10, 10))
#     correlation = np.corrcoef(fitted_classes[labels[0]]['training_points'], rowvar=False)
#     correlation[np.isnan(correlation)] = 0
#     correlation = np.abs(correlation)
#     sns.heatmap( correlation, cmap='viridis', square=True)
#     plt.title('Covariance Matrix Heatmap')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Feature Index')
#     plt.show()

training_points_split = [100, 200, 500, 10000, 30000, 50000]
validation_error_lda = []
validation_error_qda = []
class_validation_split = []


for split in training_points_split:
    gda = gaussian.Gaussian()
    gda.fit(data_train[:split], label_train[:split])

    predictions_lda = gda.predict(data_test)
    """
    Alpha, the regularization parameter to deal with PSD matrices can be optimized using cross-validation
    For QDA, is passing alpha as a dictionary for each class better ?  (not implemented)
    """
    predictions_qda = gda.predict(data_test, 'QDA')
    results_to_csv(predictions_qda.reshape(-1))
    validation_error_lda.append(zero_one_loss(label_test, predictions_lda))
    validation_error_qda.append(zero_one_loss(label_test, predictions_qda))

figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

ax1.plot(training_points_split, validation_error_lda)
ax1.title.set_text('error rate of validation set using LDA')
ax1.set_xlabel('number of training points')
ax1.set_ylabel('Error rate')

ax2.plot(training_points_split, validation_error_qda)
ax2.title.set_text('error rate of validation set using QDA')
ax2.set_xlabel('number of training points')
ax2.set_ylabel('Error rate')

plt.tight_layout()
plt.show()







