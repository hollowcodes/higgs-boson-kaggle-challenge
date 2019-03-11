import numpy as np


class nn:

	def sigmoid(self, x, deriv=False, dropout=False):

		if deriv:
			return x * (1 - x)
		s = 1 / (1 + np.exp(-x.astype("float")))

		if dropout:
			return self.dropout(s)
		else:
			return s

	def relu(self, x, deriv=False, dropout=False):

		if deriv:
			return 1/2 * (1 + np.sign(x))
		s = x/2 + 1/2 * x * np.sign(x)

		if dropout:
			return self.dropout(s)
		else:
			return s

	def mean_squared_error(self, e, deriv=False):

		if deriv:
			return 2 * e
		else:
			for i in range(len(e)):
				for k in range(len(e[0])):
					e[i][k] = pow(e[i][k], 2)

			a = np.abs(e)
			sum = 0
			for i in a:
				for j in i:
					sum += j
			return sum / len(a)

	def dropout(self, mat):

		chance = 0.5
		percent = chance * 100

		for i in range(len(mat)):
			for j in range(len(mat[0])):
				rand = np.random.randint(101)
				if rand <= percent:
					mat[i][j] *= 0
				else:
					pass
		return mat
