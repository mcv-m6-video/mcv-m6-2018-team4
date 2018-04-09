import sys
sys.path.append('../')


class KalmanFilter:

    def __init__(self, initialPosition):

        self.currentPositionX = initialPosition[0]
        self.currentPositionY = initialPosition[1]
        self.prioriEstimateX = self.currentPositionX
        self.prioriEstimateY = self.currentPositionY
        self.posterioriErrorX = 0
        self.posterioriErrorY = 0
        self.prioriErrorX = 0
        self.prioriErrorY = 0
        self.gainX = 0
        self.gainY = 0
        self.Q = 1e-5  # process variance
        self.R = 0.1 ** 2  # estimate of measurement variance, change to see effect


    def predict(self):
        return [self.prioriEstimateX, self.prioriEstimateY]

    def update(self, currentPosition):
        # Compute X update
        self.prioriErrorX = self.posterioriErrorX + self.Q
        self.gainX = self.prioriErrorX / (self.prioriErrorX + self.R)
        self.currentPositionX = self.prioriEstimateX + self.gainX * (currentPosition[0]-self.prioriEstimateX)
        self.posterioriErrorX = (1-self.gainX)*self.prioriErrorX
        self.prioriEstimateX = self.currentPositionX

        # Compute Y update
        self.prioriErrorY = self.posterioriErrorY + self.Q
        self.gainY = self.prioriErrorY / (self.prioriErrorY + self.R)
        self.currentPositionY = self.prioriEstimateY + self.gainY * (currentPosition[1]-self.prioriEstimateY)
        self.posterioriErrorY = (1-self.gainY)*self.prioriErrorY
        self.prioriEstimateY = self.currentPositionY
