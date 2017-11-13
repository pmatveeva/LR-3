import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from PIL import Image, ImageFilter
from MNIST import MNIST

# MNIST Tester class
# predict number from image
class MNISTTester(MNIST):
    def __init__(self, model_path=None, data_path=None):
        MNIST.__init__(self, model_path, data_path)

        self.init()

    def init(self):
        self.print_status('Loading a model..')

        self.init_session()

        self.load_model()

        if self.data_path is not None:
            self.load_training_data(self.data_path)

    def classify(self, feed_dict):
        number = self.sess.run(tf.argmax(self.model, 1), feed_dict)[0]
        accuracy = self.sess.run(tf.nn.softmax(self.model), feed_dict)[0]

        return number, accuracy[number]

    def predict(self, filename):
        data = self.load_image(filename)

        number, accuracy = self.classify({self.X: data})

        self.print_status('%d is %s, accuracy: %f' % (number, os.path.basename(filename), accuracy))

    def load_image(self, filename):
        img = Image.open(filename).convert('L')

        # resize to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

        # reshape -> [-1, 28, 28, 1]
        return np.reshape(data, (-1, 28, 28, 1)).tolist()
