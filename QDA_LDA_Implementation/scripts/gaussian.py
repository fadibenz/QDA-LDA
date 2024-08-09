import sys
if sys.version_info[0] < 3:
	raise Exception("Python 3 not detected.")
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import math


def singular_matrix(covariance, alpha):
    return covariance + alpha * np.identity(covariance.shape[0])


class Gaussian:
    def __init__(self):
        self.pre_calculated_metric = {}
        self.labels = None
        self.fitted_classes = {}
        self.pooled_covariance = None

    def calculate_metric(self):
        """ calculates the form found in LDA, so the classifier can evaluate test points quickly
            we do not calculate the inverse but rather solve the underlying linear system
        :return: a dictionary of the form for each class
        """
        for label in self.labels:
            metric = np.linalg.solve(self.pooled_covariance, self.fitted_classes[label]['mean'])
            self.pre_calculated_metric[label] = metric

    def fit(self, training_data, training_labels, alpha = 1e-4):
        """

        :param training_data: data point we want to train on
        :param training_labels: the true training labels
        :param alpha for the diagonal trick with singular matrices
        :return: a dictionary of the fitted gaussian parameters (mean, covariance, prior and pooled within-class covariance)
        """

        self.labels = np.unique(training_labels)
        data_length, num_features = training_data.shape
        self.pooled_covariance = np.zeros((num_features, num_features))
        for label in self.labels:

            class_training = training_data[training_labels == label]
            class_training_reshaped = class_training.reshape((class_training.shape[0], -1))
            class_length = class_training_reshaped.shape[0]

            mean = np.mean(class_training_reshaped, axis=0)
            class_prior = class_length / data_length
            unscaled_cov = np.dot((class_training_reshaped - mean).T, (class_training_reshaped - mean))

            self.pooled_covariance += unscaled_cov
            covariance = unscaled_cov / class_length

            (sign, log_det) = np.linalg.slogdet(covariance)
            if sign == 0:
                covariance = singular_matrix(covariance, alpha)
                (_, log_det) = np.linalg.slogdet(covariance)

            self.fitted_classes[label] = {
                'prior': class_prior,
                'mean': mean,
                'covariance': covariance,
                'log_det': log_det
            }

        self.pooled_covariance = self.pooled_covariance / data_length
        (sign, log_det) = np.linalg.slogdet(self.pooled_covariance)

        if sign == 0:
            self.pooled_covariance = singular_matrix(self.pooled_covariance, alpha)

        self.calculate_metric()

    def predict(self, X, discriminant_type = 'LDA'):
        if discriminant_type == 'LDA':
            conditionals = np.array([
                (self.pre_calculated_metric[class_label] @ X.T
                 - (self.pre_calculated_metric[class_label] @ self.fitted_classes[class_label]['mean']) / 2
                 + math.log(self.fitted_classes[class_label]['prior']))
                for class_label in self.labels
            ])
            return self.labels[np.argmax(conditionals, axis=0)].reshape((-1, 1))
        elif discriminant_type == 'QDA':
            n_samples, n_features = X.shape
            conditionals = np.zeros((n_samples, len(self.labels)))

            for i, class_label in enumerate(self.labels):
                centered_vector = X - self.fitted_classes[class_label]['mean']
                cov = self.fitted_classes[class_label]['covariance']

                metric = np.linalg.solve(cov, centered_vector.T).T
                mahalanobis = np.sum(metric * centered_vector, axis=1)

                conditionals[:, i] = (-0.5 * mahalanobis
                                      - 0.5 * self.fitted_classes[class_label]['log_det']
                                      + np.log(self.fitted_classes[class_label]['prior']))

            return self.labels[np.argmax(conditionals, axis=1)].reshape(-1, 1)
        else:
            raise RuntimeError('you need to specify a valid type')