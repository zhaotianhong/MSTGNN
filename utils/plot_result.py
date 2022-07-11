# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn


import matplotlib.pyplot as plt



def show_compare(y_preds, y_truths, time_step = 1, blocks = [1]):
    for b in blocks:  # b,t,n,1
        y_p = y_preds[:, time_step-1, b].reshape(-1)
        y_t = y_truths[:, time_step - 1, b].reshape(-1)
        x = [i for i in range(len(y_p))]
        plt.plot(x, y_p, label = "Prediction")
        plt.plot(x, y_t, label ="Truth")
        plt.title("SLU No. %s"%(b))
        plt.xlabel("Time")
        plt.ylabel("Bus travel demands")
        plt.legend()
        plt.show()