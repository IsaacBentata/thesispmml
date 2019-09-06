
import numpy as np

class LOSSER:
    def __init__(self):
        self.a_dim=0

    def predict(self,s,a):
        close = []
        for i, prices in enumerate(s[0]):

            if prices.shape[1] > 1:
                closes=prices[-1]
            else:
                closes = prices
            close.append(closes[-1] / closes[-2])
        weights = np.zeros(len(s[0]))
        weights[np.argmin(close)] = 1
        weights = weights[None, :]
        return weights
