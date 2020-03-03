import numpy as np
import heapq
from utils import *

class KNN:
    def __init__(self, n,distance="euclidean",normalize=True):
        self.n = n
        self.distance = distance
        self.normalize = normalize

    def euclidean_dist(self, x1, x2):
        assert len(x1) == len(x2), "dimensions of input vars differ."
        return np.sqrt(sum([(x1[i]-x2[i])**2 for i in range(len(x1))]))

    def fit(self,x,y):
        if self.normalize:
            self.mean = x.mean()
            self.std = x.std()
            self.x = normalize_to(x,self.mean,self.std)
        else:
            self.x = x
        self.y = y

    def predict_one(self,x1):
        closest = []
        for i,vector in enumerate(self.x):
            heapq.heappush(closest, (self.euclidean_dist(x1,vector),self.y[i]))
        k = self.n
        counts = {}
        while k > 0:
            label = heapq.heappop(closest)[1]
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
            k-=1
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
        return list(counts.keys())[-1]

    def predict(self,x):
        if self.normalize:
            x = normalize_to(x,self.mean,self.std)
        return [self.predict_one(v) for v in x]
