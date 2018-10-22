import os
import numpy as np
import matplotlib.pyplot as plt


# ====================================== UTILITIES =================================== #

def load_mnist(path, include_labels, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    if len(include_labels) != 0:
        label_normalize = dict(zip(include_labels, range(len(include_labels))))
        labels_mask = np.in1d(labels, include_labels)

        labels = np.array([label_normalize[label] for label in labels[labels_mask]])
        images = images[labels_mask].reshape(len(labels), 784)

    return images, labels


def train_val_split(X_train, Y_train, val_size=0.1):
    # split train set into train and val set. split proportion will be according to val_size

    num_examples = len(Y_train)
    val_indexes = np.random.choice(num_examples, int(num_examples * val_size), replace=False)

    x_val = X_train[val_indexes]
    y_val = Y_train[val_indexes]

    # now get the remain indexes for train set
    mask = np.ones(num_examples, dtype=bool)
    mask[val_indexes] = False

    x_train = X_train[mask]
    y_train = Y_train[mask]

    return x_train, x_val, y_train, y_val


def get_data():
    #  get the train,val and test sets
    path_to_fmnist_folder = './mnist_data'

    x_train, y_train = load_mnist(path_to_fmnist_folder, include_labels=[3, 5], kind='train')
    x_test, y_test = load_mnist(path_to_fmnist_folder, include_labels=[3, 5], kind='t10k')

    d = x_train.shape[1]
    num_classes = np.unique(y_train).size

    return x_train, x_test, y_train, y_test, d, num_classes


def normalize_data(x, y):
    x_train_normalized = x / 255.0
    y = y * 1.0

    return x_train_normalized, y


# ==================================================================================== #

def decision_boundary(prob):
    return 1 * (prob >= .5)


# ============================= ACTIVATION FUNCTION ================================== #

def sigmoid(X, teta):
    return 1.0 / (1 + np.exp(- np.matmul(X, teta)))


# ================================ LOSS FUNCTION ===================================== #

def cross_entropy(X, y, t):
    eps = 1e-9  # Prevent log(0)
    return - np.sum(np.multiply(y, np.log(sigmoid(X, t) + eps)) + np.multiply((1 - y), np.log(1 - sigmoid(X, t) + eps)))


# ================================ Train (GRADIENT DECENT)============================ #

def gradient(X, y, teta):
    N = X.shape[0]
    val = np.matmul(X.transpose(), sigmoid(X, teta) - y) / N
    return val


def train(x, y, eta=0.01, batch_size=20, iterations=10000):
    """ Stochastic Gradient Descent on batches of data """
    x_padded = np.hstack((x, np.ones((x.shape[0], 1))))

    feature_num = x_padded.shape[1]
    teta = np.random.normal(0, 1, feature_num)

    for iteration in range(iterations):
        batch_indices = np.random.choice(x.shape[0], batch_size, replace=False)
        # batch_indices = np.random.choice(x.shape[0], x.shape[0], replace=False)

        batch_x = x_padded[batch_indices, :]
        batch_y = y[batch_indices]

        teta = teta - eta * gradient(batch_x, batch_y, teta)

        error = np.round(cross_entropy(batch_x, batch_y, teta), 3)

        print 'Iteration = {}, Error = {}'.format(iteration, error)

    return teta


# ======================================== Test ====================================== #

def test(num_classes, classifier, x_test, y_test):
    x_padded = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    y_pred = decision_boundary(sigmoid(x_padded, classifier))

    # Create confusion matrix.
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, [y_test, y_pred], 1)

    print
    print "~== Confusion Matrix ==~"
    print cm


# ==================================================================================== #

def main():
    # Organize data
    x_train, x_test, y_train, y_test, d, num_classes = get_data()

    x_train_normalized, y_train = normalize_data(x_train, y_train)

    # Train
    classifier = train(x_train_normalized, y_train)

    # Test Classifier
    test(num_classes, classifier, x_test, y_test)


if __name__ == '__main__':
    main()
