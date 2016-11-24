#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import pickle


def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X, y, folder = [], [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    folder.append(os.path.split(subject_path)[1].decode('big5'))
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c + 1
    return [X, y, folder]


if __name__ == "__main__":
    out_dir = None
    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]"
        sys.exit()
    [X, y, names] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation
    # for thresholding:
    model = cv2.createEigenFaceRecognizer()
    model.train(np.asarray(X), np.asarray(y))
    model.save("faces.yml")
    with open("names.txt", 'wb') as f:
        pickle.dump(names, f)
    with open("y.txt", 'wb') as f:
        pickle.dump(y, f)
    # model2 = cv2.createEigenFaceRecognizer()
    # model2.load("faces.yml")FR_01.py
    # im = cv2.imread("./user3.pgm", cv2.IMREAD_GRAYSCALE)
    # [p_label, p_confidence] = model.predict(np.asarray(X[402]))
    # [p_label, p_confidence] = model.predict(im)
    # print "Predicted label = %d %s (confidence=%.2f)" % (p_label, names[y.tolist().index(p_label)], p_confidence)
    # print model.getParams()
    # # Now let's get some data:
    # mean = model.getMat("mean")
    # eigenvectors = model.getMat("eigenvectors")
    # # We'll save the mean, by first normalizing it:
    # mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    # mean_resized = mean_norm.reshape(X[0].shape)
    # if out_dir is None:
    #     cv2.imshow("mean", mean_resized)
    # else:
    #     cv2.imwrite("%s/mean.png" % (out_dir), mean_resized)
