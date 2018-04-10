
class KalmanFilter:

    def __init__(self, firstMeasurement):
        # Variables initialization

        # Prior estimate - X_k'
        self.priorEstimateX = 0
        self.priorEstimateY = 0

        # Prior errors covariance
        self.priorErrorX = 1
        self.priorErrorY = 1

        self.Q = 0.001   # process variance
        self.R = 0    # estimate of measurement variance, change to see effect

        self.update(firstMeasurement)


    def predict(self): # Time Update
        self.priorErrorX = self.posteriorErrorX + self.Q
        self.priorEstimateX = self.posteriorEstimateX

        self.priorErrorY = self.posteriorErrorY + self.Q
        self.priorEstimateY = self.posteriorEstimateY

        return [self.priorEstimateX, self.priorEstimateY]

    def update(self, measurement): # Measurement Update
        self.gainX = self.priorErrorX / (self.priorErrorX + self.R)
        self.posteriorEstimateX = self.priorEstimateX + self.gainX * (measurement[0]-self.priorEstimateX)
        # self.posteriorEstimateX = measurement[0]
        self.posteriorErrorX = (1-self.gainX)*self.priorErrorX

        self.gainY = self.priorErrorY / (self.priorErrorY + self.R)
        # self.gainY = 1.3
        self.posteriorEstimateY = self.priorEstimateY + self.gainY * (measurement[1]-self.priorEstimateY)
        # self.posteriorEstimateY = measurement[1]
        self.posteriorErrorY = (1-self.gainY)*self.priorErrorY

