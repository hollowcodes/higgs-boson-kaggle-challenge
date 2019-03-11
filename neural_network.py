from higgs_data import preprocessing
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from utils import nn

nn = nn()

class preprocess:

		def __init__(self):

			self.data = preprocessing(amount=1000).load_higgs()

			random.shuffle(self.data)

			amount = len(self.data)

			samples, labels = [], []
			for i in self.data:
				samples.append(i[0])
				labels.append(i[1])

			test_size = 0.30

			self.train_x, self.test_x, self.train_y, self.test_y =  samples[int(amount * test_size):], samples[:int(amount * test_size)], \
																	labels[int(amount * test_size):], labels[:int(amount * test_size):]

		def change_dimensions(self, data):

			d = []
			for i in data:
				d.append(np.asarray([i]).T)

			return d

		def dataset(self):

			train_x, test_x, train_y, test_y = self.change_dimensions(self.train_x), self.change_dimensions(self.test_x), self.train_y, self.test_y
			
			return train_x, test_x, train_y, test_y


class train_nn:

	def __init__(self):

		self.hiddensize = 25
		self.b = 1
		self.lr = 0.01
		self.epochs = 250

		self.train_x, self.test_x, self.train_y, self.test_y = preprocess().dataset()

		self.syn0 = 2 * np.random.random((self.hiddensize, len(self.train_x[0]))) - 1
		self.syn1 = 2 * np.random.random((len(self.train_y[0]), self.hiddensize)) - 1

		self.losses = []

	def model(self):

		print("\n[ training with ", len(self.train_x), " samples ]\n")

		for epoch in range(self.epochs):

			mean_loss = []
			for i in range(len(self.train_x)):

				X = np.asarray(self.train_x[i])
				y = np.asarray(self.train_y[i])

				l0 = X

				l1 = nn.relu(np.dot(self.syn0, l0) + self.b, deriv=False, dropout=True)

				l2 = nn.sigmoid(np.dot(self.syn1, l1) + self.b, deriv=False, dropout=False)

				l2_error = np.subtract(l2, y)
				l2_delta = nn.mean_squared_error(l2_error, True) * nn.sigmoid(l2, deriv=True, dropout=False)

				l1_error = np.dot(l2_delta.T, self.syn1)
				l1_delta = nn.mean_squared_error(l1_error.T, True) * nn.relu(l1, deriv=True, dropout=False)

				self.syn0 -= self.lr * np.dot(l0.astype("float"), l1_delta.T.astype("float")).T
				self.syn1 -= self.lr * np.dot(l1.astype("float"), l2_delta.T.astype("float")).T

				mean_loss.append(nn.mean_squared_error(l2_error))

			print("loss after epoch [ " + str(epoch + 1) + " /", self.epochs, "] :", np.mean(mean_loss))
			self.losses.append(np.mean(mean_loss))

	def test(self):

		test_nn(self.syn0, self.syn1, self.b, self.test_x, self.test_y).test_nn()

	def plot_loss(self):

		ys = []
		for i in range(len(self.losses)):
			ys.append(i)

		plt.style.use("ggplot")

		plt.plot(ys, self.losses, "r--")
		plt.title("loss curve")
		plt.show()


class test_nn:

	def __init__(self, syn0, syn1, b, test_x, test_y):

		self.syn0 = syn0
		self.syn1 = syn1
		self.b = b

		self.test_x, self.test_y = test_x, test_y

	def check(self, output, expected):

		for i in range(len(output)):
			output[i] = output[i].round(0)

		if list(output) == list(expected):
			return 1
		else:
			return 0

	def test_nn(self):

		print("\n[ testing with ", len(self.test_x), " samples ]\n")

		correct_classified = 0
		testing_iterations = len(self.test_x)

		for i in range(testing_iterations):

			l1 = nn.relu(np.dot(self.syn0, self.test_x[i]) + self.b, deriv=False, dropout=False)
			l2 = nn.sigmoid(np.dot(self.syn1, l1) + self.b, deriv=False, dropout=False)

			print("testing sample [ ", i, "/", testing_iterations, " ]")

			correct_classified += self.check(l2, self.test_y[i])

		print("\ntesting with ", testing_iterations, " samples\ncorrect classified: ", correct_classified, " -> ", round((correct_classified / testing_iterations) * 100, 4), "%")


train_network = train_nn()

train_network.model()
train_network.test()
train_network.plot_loss()

# preprocess().dataset()
