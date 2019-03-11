import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
import random


class preprocessing:

	def __init__(self, amount):

		self.amount = amount

		self.higgs_data = pd.read_csv("C:\\Users\\wndws10tp.DESKTOP-TTE2SIQ\\PycharmProjects\\neuraln\\higgs_boson_challenge\\training.csv").values

	def normalize(self):

		# deleting id
		higgs_data = []
		for i in self.higgs_data[:self.amount]:
			higgs_data.append(np.delete(i, 0))

		# normalize [0; 1]
		maxima = []
		minima = []
		for i in range(len(higgs_data[0]) - 1):
			lst = []
			for j in range(len(higgs_data)):
				lst.append(higgs_data[j][i])
			maxima.append(max(lst))
			minima.append(min(lst))

		for i in range(len(higgs_data)):
			for j in range(len(higgs_data[0]) - 1):
				higgs_data[i][j] = round((higgs_data[i][j] - minima[j]) / (maxima[j] - minima[j]), 7)	# z_i = (x_i - min(x)) / (max(x) - min(x))

		return higgs_data

	def shape(self):

		higgs_data = self.normalize()

		# replacing label
		for i in range(len(higgs_data)):
			# signal
			if higgs_data[i][-1] == "s":
				higgs_data[i][-1] = [[1], [0]]
			# background
			elif higgs_data[i][-1] == "b":
				higgs_data[i][-1] = [[0], [1]]

		dataset = []
		for i in higgs_data:
			samples = i[:-1]
			label = i[-1]
			row = [samples, label]
			dataset.append(row)

		return dataset

	def load_higgs(self):

		return self.shape()

	