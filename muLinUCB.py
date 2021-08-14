import numpy as np

def fillThetaContext(layerInfo, theta_context_dim):
    Action_num = len(layerInfo)
    x_theta = np.zeros((theta_context_dim, Action_num))
    actionList = []
    for i in range(Action_num):
        x_theta[0][i] = layerInfo[i][3]
        x_theta[1][i] = layerInfo[i][0]

        x_theta[2][i] = layerInfo[i][4]
        x_theta[3][i] = layerInfo[i][1]

        x_theta[4][i] = layerInfo[i][5]
        x_theta[5][i] = layerInfo[i][2]

        x_theta[6][i] = layerInfo[i][6]
        actionList.append(layerInfo[i][7])
    return x_theta, actionList

def getCx(x_theta, Action_num):
    listC_x = []
    for i in range(Action_num):
        temp = np.sqrt(np.matmul(x_theta[:, [i]].T, x_theta[:, [i]]))
        listC_x.append(temp[0][0])
    Cx = pow(max(listC_x), 2)
    return Cx

class muLinUCB():
    def __init__(self, mu, layerInfo, frontDelay):
        self.mu = mu
        self.numOfAction = len(layerInfo)
        self.thetaContextDim = 7
        self.x_theta, self.actionList = fillThetaContext(layerInfo, self.thetaContextDim)
        self.C_x = getCx(self.x_theta, self.numOfAction)
        self.frontDelay = frontDelay

        self.frameNum = 200
        self.delta = 0.1
        self.C_noise = 0.05
        self.l_key = 0.8
        self.l_nonkey = 0.2
        self.C_theta = 1
        self.A = np.diag(np.random.randint(1, 9, size=self.thetaContextDim))
        self.b = np.zeros((self.thetaContextDim, 1))
        self.alpha = (self.C_theta + np.sqrt(np.log((1 + self.frameNum * self.C_x * self.C_x)/self.delta) * self.thetaContextDim)*self.C_noise)/(1 - self.l_key)

        self.forceSamplingRate = 0.25
        self.forceSampleFrame = np.ceil(np.power(self.frameNum, self.forceSamplingRate))
        print('forceSampleFrame:', self.forceSampleFrame)

    def updateDoublingTrickFrameNum(self, current_frame):
        if current_frame > self.frameNum:
            self.frameNum = self.frameNum * 2
            self.alpha = (self.C_theta + np.sqrt(np.log((1 + self.frameNum * self.C_x * self.C_x) / self.delta) * self.thetaContextDim) * self.C_noise) / (1 - self.l_key)
            self.forceSampleFrame = np.ceil(np.power(self.frameNum, self.forceSamplingRate))
            return True
        return False

    def getEstimationAction(self, key_frame, current_frame):
        A_inv = np.linalg.inv(self.A)
        theta = np.matmul(A_inv, self.b)

        if key_frame:
            L = self.l_key
        else:
            L = self.l_nonkey

        estimate_delay = []

        for action_index in range(self.numOfAction):
            x_1 = np.copy(self.x_theta[:, [action_index]])
            x_2 = np.copy(self.x_theta[:, [action_index]])

            temp_1 = np.matmul(x_1.T, theta)
            temp_2 = self.alpha * np.sqrt((1 - L) * np.matmul(np.matmul(x_1.T, A_inv), x_2))

            estimate_delay.append(temp_1 - temp_2 + self.frontDelay[action_index])

        if current_frame % self.forceSampleFrame == 0:
            estimate_action = estimate_delay.index(min(estimate_delay[0:-1]))
        else:
            estimate_action = estimate_delay.index(min(estimate_delay))
        return estimate_action

    def updateA_b(self, estimate_action, actual_delay):
        if estimate_action != self.numOfAction - 1:
            self.A = self.A + np.matmul(self.x_theta[:, [estimate_action]], self.x_theta[:, [estimate_action]].T)
            self.b = self.b + self.x_theta[:, [estimate_action]] * actual_delay


if __name__ == '__main__':
    partitionInfo = {
        0: [13, 3, 24, 15346630656, 123633664, 26208256, 4818272],
        1: [12, 3, 23, 15259926528, 123633664, 22996992, 102761824],
        2: [11, 3, 22, 13410238464, 123633664, 19785728, 102761824],
        3: [11, 3, 21, 13410238464, 123633664, 16574464, 25691488],
        4: [10, 3, 20, 12485394432, 123633664, 13363200, 51381600],
        5: [9, 3, 19, 10635706368, 123633664, 10151936, 51381600],
        6: [9, 3, 18, 10635706368, 123633664, 8546304, 12846432],
        7: [8, 3, 17, 9710862336, 123633664, 6940672, 25691496],
        8: [7, 3, 16, 7861174272, 123633664, 5335040, 25691496],
        9: [6, 3, 15, 6011486208, 123633664, 4532224, 25691496],
        10: [6, 3, 14, 6011486208, 123633664, 3729408, 6423912],
        11: [5, 3, 13, 5086642176, 123633664, 2926592, 12846440],
        12: [4, 3, 12, 3236954112, 123633664, 2123776, 12846440],
        13: [3, 3, 11, 1387266048, 123633664, 1320960, 12846440],
        14: [3, 3, 10, 1387266048, 123633664, 919552, 3212648],
        15: [2, 3, 9, 924844032, 123633664, 518144, 3212648],
        16: [1, 3, 8, 462422016, 123633664, 417792, 3212648],
        17: [0, 3, 7, 0, 123633664, 317440, 3212648],
        18: [0, 3, 6, 0, 123633664, 217088, 3212648],
        19: [0, 3, 4, 0, 123633664, 16384, 804200],
        20: [0, 2, 2, 0, 20873216, 12288, 804200],
        21: [0, 1, 0, 0, 4096000, 0, 132416],
        22: [0, 0, 0, 0, 0, 0, 0]
    }

    frontDelay = [0 for index in range(len(partitionInfo))]
    muLinUCB = muLinUCB(0.25, partitionInfo, frontDelay)











