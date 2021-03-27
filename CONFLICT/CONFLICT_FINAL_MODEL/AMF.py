import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp

dB = 3
dP = 5
aucStep = 4


class export:
    def __init__(self, etaLatent, etaRowBias, etaLatentScaler, etaPair, etaBilinear, folds, epochs, CVSet, testSet):

        #learning rates
        self.etaLatent = etaLatent
        self.etaRowBias = etaRowBias
        self.etaLatentScaler = etaLatentScaler
        self.etaPair = etaPair
        self.etaBilinear = etaBilinear

        #training settings
        self.folds = folds
        self.epochs = epochs
        #set
        self.CVSet = CVSet
        self.testSet = testSet


class exportModel:
    def __init__(self, etaLatent, etaRowBias, etaLatentScaler, etaBilinear, epochs, alpha, epsilon, U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, x0HR, y0HR, xAHR, yAHR):
        # learning rates
        self.etaLatent = etaLatent
        self.etaRowBias = etaRowBias
        self.etaLatentScaler = etaLatentScaler
        self.etaBilinear = etaBilinear

        # training settings
        self.epochs = epochs
        self.alpha = alpha
        self.epsilon = epsilon

        #trained model
        self.U = U
        self.ULatentScaler = ULatentScaler
        self.UBias = UBias
        self.WBilinear = WBilinear

        # adversarially trained model
        self.UAdv = UAdv
        self.ULatentScalerAdv = ULatentScalerAdv
        self.UBiasAdv = UBiasAdv
        self.WBilinearAdv = WBilinearAdv

        # data HR
        self.x0HR = x0HR
        self.y0HR = y0HR
        self.xAHR = xAHR
        self.yAHR = yAHR


class weight:
    def __init__(self, p, n):
        self.U = np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[p,n]).astype(np.float128))  # for k latent features and m nodes
        self.UBias = np.random.choice([-0.000000000000000000000000001, 0.000000000000000000000000000001], size=[n]).astype(np.float128)
        self.ULatentScaler = np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[p, p]).astype(np.float128))  # for asymmetric; for symmetric, use diag(randn(k)) instead
        self.WBilinear = np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[dB, dB]).astype(np.float128)) # for dBilinear features for each node


class lambdas:
    def __init__(self):
        self.lambdaLatent = [0]  # regularization for node's latent vector U
        self.lambdaRowBias = [0] # regularization for node's bias UBias
        self.lambdaLatentScaler = [0]  # regularization for scaling factors Lambda (in paper)
        self.lambdaBilinear = [0]  # regularization for weights on node features
        self.lambdaScaler = [0] # scaling factor for regularization, can be set to 1 by default


class eta:
    def __init__(self):
        self.etaLatent = [0.01]#[0.01, 0.1, 0.5]  # learning rate for latent feature
        self.etaRowBias = [0.005]#[0.001, 0.01, 0.1]  # learning rate for node bias
        self.etaLatentScaler = [0.05]#[0.01, 0.1, 0.5]  # learning rate for scaler to latent features
        self.etaPair = [0]
        self.etaBilinear = [0.05]#[0.01, 0.1, 0.5]
        self.etaBias = 0.000005 # learning rate for global bias, used when features are present
        self.epochs = [1000]#[250, 500, 1000] # # of passes of SGD
        self.epsilon = [0.5, 1]
        self.alpha = [0.5, 1]


def sgd(D, sideBilinear, weights, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Bilinear, epochs, plot0, plot0HR, validationSet):

    pairs = len(D[1, :])
    testOnes = []
    for index in range(0, len(validationSet[0, :])):
        if validationSet[2, index] == 1.0:
            testOnes.append(index)

    #Load model parameters into sgd
    U = np.copy(weights.U)
    UBias = np.copy(weights.UBias)
    ULatentScaler = np.copy(weights.ULatentScaler)
    WBilinear = np.copy(weights.WBilinear)

    #load initial learning rates
    # Dampening of the learning rates across epochs (optional)
    etaLatent = eta_Latent  # / ((1 + etaLatent0 * lambdaLatent) * e)
    etaRowBias = eta_RowBias  # / ((1 + etaRowBias0 * lambdaRowBias) * e)
    etaLatentScaler = eta_LatentScaler  # / ((1 + etaLatentScaler0 * lambdaLatentScaler) * e)
    etaBilinear = eta_Bilinear  # / ((1 + etaBilinear0 * lambdaBilinear) * e)

    #set the counter for the plotting (aucCounter is used both to plot AUC and HitRatio)
    aucCounter = 0
    limit = int(1 * pairs)

    # Main SGD body
    for e in range(1, epochs+1):

        for t in range(0, limit):
            i = int(D[0, t]) - 1
            j = int(D[1, t]) - 1
            truth = int(D[2, t])

            prediction = U[:, i].T @ ULatentScaler @ U[:, j] + UBias[i] + UBias[j] + sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j]
            sigma = 1 / (1 + np.exp(-prediction))  # this must be a matrix of link probabilities.

            #cost = -(truth*np.log(sigma)+(1-truth)*(np.log(1-sigma)))


            #gradients computation
            gradscaler = float(sigma - truth)
            gradI = ULatentScaler @ U[:, j]
            gradJ = (U[:, i].T @ ULatentScaler).T
            gradRowBias = 1  # ones(1, numel(examples))        # 1 x 1
            gradLatentScaler = U[:, i] @ U[:, j].T
            gradBilinear = sideBilinear[:, i] @ sideBilinear[:, j].T

            #update section
            U[:, i] = U[:, i] - etaLatent * (gradscaler * gradI)  # + lambdaLatent * lambdaScaler * U[:, i])  # U_i è di dimensione 5x1
            U[:, j] = U[:, j] - etaLatent * (gradscaler * gradJ)  # + lambdaLatent * lambdaScaler * U[:, j])
            UBias[i] = UBias[i] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[i])
            UBias[j] = UBias[j] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[j])
            ULatentScaler = ULatentScaler - etaLatentScaler * (gradscaler * gradLatentScaler)  # + lambdaLatentScaler * ULatentScaler)
            WBilinear = WBilinear - etaBilinear * (gradscaler * gradBilinear)  # + lambdaBilinear * WBilinear)


        if e % aucStep == 0:
            print("Test")
            prediction = predict(U, UBias, ULatentScaler, WBilinear, sideBilinear)
            acc, rec, prec, auc, f1 = test(prediction, validationSet)
            hr = hitratio(U, ULatentScaler, UBias, WBilinear, validationSet, testOnes)
            plot0[aucCounter, 0] = auc
            plot0HR[aucCounter, 0] = hr
            aucCounter += 1

    return U, UBias, ULatentScaler, WBilinear, plot0, plot0HR, aucCounter


def sgd_adv(D, sideBilinear, U0, UBias0, ULatentScaler0, WBilinear0, eta_Latent, eta_LatentScaler, eta_Bilinear, eta_RowBias, epochs, plotAdv, plotAdvHR, validationSet, alpha,  epsilon):

    pairs = len(D[1, :])
    testOnes = []

    for index in range(0, len(validationSet[0, :])):
        if validationSet[2, index] == 1.0:
            testOnes.append(index)

    #load model parameter, passed from the initial training SGD function
    U = np.copy(U0)
    UBias = np.copy(UBias0)
    ULatentScaler = np.copy(ULatentScaler0)
    WBilinear = np.copy(WBilinear0)


    #load starting learning rates
    etaLatent = eta_Latent  # / ((1 + etaLatent0 * lambdaLatent) * e)
    etaRowBias = eta_RowBias  # / ((1 + etaRowBias0 * lambdaRowBias) * e)
    etaLatentScaler = eta_LatentScaler  # / ((1 + etaLatentScaler0 * lambdaLatentScaler) * e)
    etaBilinear = eta_Bilinear  # / ((1 + etaBilinear0 * lambdaBilinear) * e)

    limit = int(1 * pairs)

    #initalize perturbations
    DeltaI = np.zeros(len(U[:, 0]))
    DeltaJ = np.zeros(len(U[:, 0]))
    DeltaXI = np.zeros(len(sideBilinear[:, 0]))
    DeltaXJ = np.zeros(len(sideBilinear[:, 0]))

    # np.sqrt(np.power(epsilon, 2)/len(U[:, 0]))

    # here we set the bound (if starting perturbation is random)for the random noise that could be 0 if we want to add Delta=0 or the commented value if we want ||Delta||<epsilon

    aucCounter = 0

    # Main SGD body
    for e in range(1, epochs):
        for t in range(0, limit):
            i = int(D[0, t]) - 1
            j = int(D[1, t]) - 1
            truth = int(D[2, t])

            # Procedure for the adversarial perturbation buidling
            predictionDelta = (U[:, i].T+DeltaI) @ ULatentScaler @ (U[:, j].T + DeltaJ).T + UBias[i] + UBias[j] + (sideBilinear[:, i].T + DeltaXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaXJ).T # WPair @ sidePair[:, i, j]  # + WBias

            sigmaDelta = 1 / (1 + np.exp(-predictionDelta))  # this must be a matrix of link probabilities.

            GammaI = alpha * (sigmaDelta - truth) * (ULatentScaler @ (U[:, j].T+DeltaJ).T).T
            GammaJ = alpha * (sigmaDelta - truth) * ((U[:, i].T+DeltaI) @ ULatentScaler)
            GammaXI = alpha * (sigmaDelta - truth) * (WBilinear @ (sideBilinear[:, j].T+DeltaXJ).T).T
            GammaXJ = alpha * (sigmaDelta - truth) * ((sideBilinear[:, i].T+DeltaXI) @ WBilinear)

            DeltaAdvI = epsilon * GammaI / np.sqrt(max(np.sum(np.power(GammaI, 2)), 0.000001))
            DeltaAdvJ = epsilon * GammaJ / np.sqrt(max(np.sum(np.power(GammaJ, 2)), 0.000001))
            DeltaAdvXI = epsilon * GammaXI / np.sqrt(max(np.sum(np.power(GammaXI, 2)), 0.000001))
            DeltaAdvXJ = epsilon * GammaXJ / np.sqrt(max(np.sum(np.power(GammaXJ, 2)), 0.000001))

            predictionAdv = (U[:, i].T + DeltaAdvI) @ ULatentScaler @ (U[:, j].T + DeltaAdvJ).T + UBias[i] + UBias[j] + \
                            (sideBilinear[:, i].T + DeltaAdvXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaAdvXJ).T #+ WPair @ sidePair[:, i, j]
            prediction = (U[:, i]).T @ ULatentScaler @ U[:, j] + UBias[i] + UBias[j] + sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j]

            sigmaAdv = 1 / (1 + np.exp(-predictionAdv))
            sigma = 1 / (1 + np.exp(-prediction))

            #cost = -(truth * np.log(sigma) + (1 - truth) * (np.log(1-sigma))) - alpha * ((truth * np.log(sigmaAdv)) + (1 - truth) * (np.log(1-sigmaAdv)))

            gradscalerAdv = float(sigmaAdv - truth)
            gradscaler = float(sigma - truth)

            #gradients computation
            gradIAdv = ULatentScaler @ (U[:, j].T + DeltaAdvJ).T  # + np.dot(ULatentScalerAdv, np.transpose(DeltaAdvJ))
            gradI = ULatentScaler @ U[:, j]
            gradJAdv = ((U[:, i].T + DeltaAdvI) @ ULatentScaler).T
            gradJ = (U[:, i].T @ ULatentScaler).T
            gradBilinear = sideBilinear[:, i] @ sideBilinear[:, j].T
            gradBilinearAdv = sideBilinear[:, i] @ sideBilinear[:, j].T + sideBilinear[:, j] @ DeltaAdvXI + sideBilinear[:, i] @ DeltaAdvXJ + DeltaAdvXI @ DeltaAdvXJ.T
            gradBias = 1
            gradLatentScaler = U[:, i] @ U[:, j].T
            gradLatentScalerAdv = U[:, i] @ U[:, j].T + U[:, j] @ DeltaAdvI + U[:, i] @ DeltaAdvJ + DeltaAdvI @ DeltaAdvJ.T

            #updates
            U[:, i] = U[:, i] - etaLatent * (gradscaler * gradI + alpha * gradscalerAdv * gradIAdv)  # U_i è di dimensione 2x1
            U[:, j] = U[:, j] - etaLatent * (gradscaler * gradJ + alpha * gradscalerAdv * gradJAdv)
            UBias[i] = UBias[i] - etaRowBias * (gradscaler * gradBias)
            UBias[j] = UBias[j] - etaRowBias * (gradscaler * gradBias)
            ULatentScaler = ULatentScaler - etaLatentScaler * (gradscaler * gradLatentScaler + alpha * gradscalerAdv * gradLatentScalerAdv)
            WBilinear = WBilinear - etaBilinear * (gradscaler * gradBilinear + alpha * gradscalerAdv * gradBilinearAdv)

        if e % aucStep == 0:
            prediction = predict(U, UBias, ULatentScaler, WBilinear, sideBilinear)
            acc, rec, prec, auc, f1 = test(prediction, validationSet)
            hr = hitratio(U, ULatentScaler, UBias, WBilinear, validationSet, testOnes)
            plotAdvHR[aucCounter, 0] = hr
            plotAdv[aucCounter, 0] = auc
            aucCounter += 1

    return U, UBias, ULatentScaler, WBilinear, plotAdv, plotAdvHR


def sgd_continue(D, sideBilinear, U, UBiasContinue, ULatentScaler, WBilinear, eta_Latent, eta_LatentScaler, eta_Bilinear, eta_RowBias, epochs, plot0, plot0HR, validationSet, plotPoint):

    pairs = len(D[1, :])
    testOnes = []

    for index in range(0, len(validationSet[0, :])):
        if validationSet[2, index] == 1.0:
            testOnes.append(index)

    U = np.copy(U)
    UBias = np.copy(UBiasContinue)
    ULatentScaler = np.copy(ULatentScaler)
    WBilinear = np.copy(WBilinear)

    etaLatent = eta_Latent  # / ((1 + etaLatent0 * lambdaLatent) * e)
    etaRowBias = eta_RowBias  # / ((1 + etaRowBias0 * lambdaRowBias) * e)
    etaLatentScaler = eta_LatentScaler  # / ((1 + etaLatentScaler0 * lambdaLatentScaler) * e)
    etaBilinear = eta_Bilinear  # / ((1 + etaBilinear0 * lambdaBilinear) * e)

    limit = int(1 * pairs)

    aucCounter = plotPoint

    # Main SGD body
    for e in range(1, epochs):
        for t in range(0, limit):
            i = int(D[0, t]) - 1
            j = int(D[1, t]) - 1
            truth = int(D[2, t])

            prediction = U[:, i].T @ ULatentScaler @ U[:, j] + UBias[i] + UBias[j] + sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j]

            sigma = 1 / (1 + np.exp(-prediction))

            #cost = -(truth*np.log(sigma)+(1-truth)*(np.log(1-sigma)))

            # Common gradient scaler
            gradscaler = float(sigma - truth)

            gradI = ULatentScaler @ U[:, j]
            gradJ = (U[:, i].T @ ULatentScaler).T
            gradRowBias = 1  # ones(1, numel(examples))        # 1 x 1
            gradLatentScaler = U[:, i] @ U[:, j].T
            gradBilinear = sideBilinear[:, i] @ sideBilinear[:, j].T

            #updates
            U[:, i] = U[:, i] - etaLatent * (gradscaler * gradI)  # + lambdaLatent * lambdaScaler * U[:, i])  # U_i è di dimensione 5x1
            U[:, j] = U[:, j] - etaLatent * (gradscaler * gradJ)  # + lambdaLatent * lambdaScaler * U[:, j])
            UBias[i] = UBias[i] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[i])
            UBias[j] = UBias[j] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[j])
            ULatentScaler = ULatentScaler - etaLatentScaler * (gradscaler * gradLatentScaler)  # + lambdaLatentScaler * ULatentScaler)
            WBilinear = WBilinear - etaBilinear * (gradscaler * gradBilinear)  # + lambdaBilinear * WBilinear)

        if e % aucStep == 0:
            prediction = predict(U, UBias, ULatentScaler, WBilinear, sideBilinear)
            acc, rec, prec, auc, fot1 = test(prediction, validationSet)
            plot0[aucCounter, 0] = auc
            hr = hitratio(U, ULatentScaler, UBias, WBilinear, validationSet, testOnes)
            plot0HR[aucCounter, 0] = hr
            aucCounter += 1

    return U, UBias, ULatentScaler, WBilinear, plot0, plot0HR


def hitratio(U, ULatentScaler, UBias, WBilinear, test, testOnes):

    #test: [3xn]: row1=nodo1 |row2=nodo2 | row3=0/1
    #testOnes: vector containing indexer about positive test samples (samples with raw3=1)

    divider = len(testOnes)
    hr = 0
    predictionNode = np.random.rand(len(U[0, :]), 2)
    indexes = np.arange(1, (len(U[0, :]) + 1))
    predictionNode[:, 1] = np.copy(indexes)
    limit = len(U[0, :])
    for node in range(0, limit):
        prediction = U[:, node].T @ ULatentScaler @ U + UBias[node] + UBias + sideBilinear[:, node] @ WBilinear @ sideBilinear
        predictionNode[:, 0] = np.copy(prediction)
        predictionNode = predictionNode[np.argsort(predictionNode[:, 0])[::-1]]

        for ii in range(0, 10):
            for jj in testOnes:
                if int(test[0, jj]) == node and test[1, jj] == predictionNode[ii, 1]:
                    hr += 1

    hr = hr / divider
    return hr


def master(D, expCounter, sideBilinear, weightClass, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Bilinear, epochs, alpha, epsilon, testSet):

    v_training_set = []  # set of indexes of the D where exists and edge
    testOnes = []

    for index in range(0, len(testSet[0, :])):
        if testSet[2, index] == 1.0:
            testOnes.append(index)

    for index in range(0, len(D[0, :])):
        if D[2, index] == 1.0:
            v_training_set.append(index)

    accuracy = []
    precision = []
    recall = []
    AUC = []
    F1 = []
    HR = []

    accuracyAdv = []
    precisionAdv = []
    recallAdv = []
    AUCAdv = []
    F1Adv = []
    HRAdv = []

    # newepochs is the number of epochs for the sgd_adv learning
    if epochs % 2 == 0:
        Newepochs = int(epochs / 2)
    else:
        Newepochs = int((epochs + 1) / 2)

    # plot parameters
    aucSamples = int(math.ceil((epochs+Newepochs)/aucStep))
    aucSamplesAdv = int(math.ceil(Newepochs/aucStep))
    plot0 = np.random.rand(aucSamples-1, 1)
    plotAdv = np.random.rand(aucSamplesAdv-1, 1)
    plot0final = np.random.rand(aucSamples-1)
    plotAdvfinal = np.random.rand(aucSamplesAdv-1)

    plot0HR = np.random.rand(aucSamples-1, 1)
    plotAdvHR = np.random.rand(aucSamplesAdv-1, 1)
    plot0finalHR = np.random.rand(aucSamples-1)
    plotAdvfinalHR = np.random.rand(aucSamplesAdv-1)

    trainSet = np.copy(D[:, :])
    testSet = np.copy(TestSet[:, :])

    U, UBias, ULatentScaler, WBilinear, plot0, plot0HR, plotPoint = sgd(trainSet, sideBilinear, weightClass, eta_Latent,
                                                                        eta_RowBias, eta_LatentScaler, eta_Bilinear,
                                                                        epochs, plot0, plot0HR, testSet)

    UAdv, UBiasAdv, ULatentScalerAdv, WBilinearAdv, plotAdv, plotAdvHR = sgd_adv(trainSet, sideBilinear, U, UBias,
                                                                                 ULatentScaler, WBilinear, eta_Latent,
                                                                                 eta_LatentScaler, eta_Bilinear,
                                                                                 eta_RowBias, Newepochs, plotAdv,
                                                                                 plotAdvHR, testSet, alpha,
                                                                                 epsilon)

    UCon, UBiasCon, ULatentScalerCon, WBilinearCon, plot0, plot0HR = sgd_continue(trainSet, sideBilinear, U, UBias,
                                                                                  ULatentScaler, WBilinear, eta_Latent,
                                                                                  eta_LatentScaler, eta_Bilinear,
                                                                                  eta_RowBias, Newepochs, plot0,
                                                                                  plot0HR, testSet, plotPoint)

    predictionAdv = predict(UAdv, UBiasAdv, ULatentScalerAdv, WBilinearAdv, sideBilinear)  # generate prediction matrix
    predictionCon = predict(UCon, UBiasCon, ULatentScalerCon, WBilinearCon, sideBilinear)  # generate prediction matrix

    accAdv, recAdv, precAdv, aucAdv, f1Adv = test(predictionAdv, testSet)
    accCon, recCon, precCon, aucCon, f1Con = test(predictionCon, testSet)

    testOnes = []

    for index in range(0, len(testSet[0, :])):
        if testSet[2, index] == 1.0:
            testOnes.append(index)

    hrAdv = hitratio(UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, testSet, testOnes)
    hrCon = hitratio(UCon, ULatentScalerCon, UBiasCon, WBilinearCon, testSet, testOnes)

    accuracyAdv.append(accAdv)
    recallAdv.append(recAdv)
    precisionAdv.append(precAdv)
    AUCAdv.append(aucAdv)
    F1Adv.append(f1Adv)
    HRAdv.append(hrAdv)

    accuracy.append(accCon)
    recall.append(recCon)
    precision.append(precCon)
    AUC.append(aucCon)
    F1.append(f1Con)
    HR.append(hrCon)

    for i in range(0, len(plot0[:, 0])):
        plot0final[i] = np.mean(plot0[i, :])
        plot0finalHR[i] = np.mean(plot0HR[i, :])
    for i in range(0, len(plotAdv[:,0])):
        plotAdvfinal[i] = np.mean(plotAdv[i, :])
        plotAdvfinalHR[i] = np.mean(plotAdvHR[i, :])

    # Plot the AUC
    plotString = str(expCounter) + '_AUC_plot.jpg' #'./exportCurves/' +
    x0 = np.arange(0, len(plot0[:, 0]))
    x0 = aucStep * (x0+1)
    y0 = plot0final
    xA = np.arange(0, len(plotAdv[:, 0]))
    xA = (aucStep * (xA+1)) + epochs
    yA = plotAdvfinal
    fig = plt.figure(figsize=(10,5))
    plt.plot(x0, y0, label="MF")
    plt.plot(xA, yA, label="AMF")
    # Labeling the X-axis
    plt.xlabel('Iterations')
    # Labeling the Y-axis
    plt.ylabel('AUC')
    plt.title('AUC trend comparison')
    plt.legend()
    plt.ylim(bottom=0, top=1)
    #Saving the plot as an image
    fig.savefig(plotString, bbox_inches='tight', dpi=150)

    # Plot the HR
    plotStringHR = str(expCounter) + '_HR_plot.jpg' #'./exportCurves/' +
    x0HR = np.arange(0, len(plot0HR[:, 0]))
    x0HR = aucStep * (x0HR+1)
    y0HR = plot0finalHR
    xAHR = np.arange(0, len(plotAdvHR[:, 0]))
    xAHR = (aucStep * (xAHR+1)) + epochs
    yAHR = plotAdvfinalHR
    fig = plt.figure(figsize=(10,5))
    plt.plot(x0HR, y0HR, label="MF")
    plt.plot(xAHR, yAHR, label="AMF")
    # Labeling the X-axis
    plt.xlabel('Iterations')
    # Labeling the Y-axis
    plt.ylabel('HR@10')
    plt.title('Hit Ratio trend comparison')
    plt.legend()
    plt.ylim(bottom=0, top=1)
    #Saving the plot as an image
    fig.savefig(plotStringHR, bbox_inches='tight', dpi=150)

    print("MODEL: ", expCounter)
    print("eta_Latent: ", eta_Latent)
    print("eta_RowBias: ", eta_RowBias)
    print("eta_LatentScaler: ", eta_LatentScaler)
    print("eta_Bilinear: ", eta_Bilinear)
    print("Epochs per Kfold:", (epochs - 1))
    print("-----------------------------------")
    print("     METRICS for MODEL: ", expCounter)
    print("--- Matrix Factorization ----")
    print("Accuracy:", accCon)
    print("Precision:", precCon)
    print("Recall:", recCon)
    print("F1:",f1Con)
    print("AUC:", aucCon)
    print("HR@10:",hrCon)
    print("---- Adversarial Matrix Factorization ----")
    print("Accuracy:", accAdv)
    print("Precision:",precAdv)
    print("Recall:", recAdv)
    print("F1:", f1Adv)
    print("AUC:", aucAdv)
    print("HR@10:", hrAdv)

    expString = './exportFiles/' + str(expCounter) + '_file.pkl'
    exportObject = exportModel(eta_Latent, eta_RowBias, eta_LatentScaler, eta_Bilinear, (epochs - 1), alpha, epsilon, UCon, ULatentScalerCon, UBiasCon, WBilinearCon, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, x0HR, y0HR, xAHR, yAHR)
    f = open(expString, 'wb')
    pickle.dump(exportObject, f, -1)
    f.close()

    return


def test(prediction, groundTruth):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    prediction_01 = np.random.rand(len(prediction[:, 0]), len(prediction[:, 0]))

    for i in range(0, len(prediction[0, :])):
        for j in range(0, len(prediction[:, 0])):
            if prediction[i, j] < 0.5:
                prediction_01[i, j] = 0.0
            else:
                prediction_01[i, j] = 1.0

    prediction_auc = np.random.rand(len(groundTruth[:, 0]), len(groundTruth[0, :]))

    for t in range(0, len(groundTruth[0, :])):
        i = int(groundTruth[0, t]) - 1
        j = int(groundTruth[1, t]) - 1
        truth = groundTruth[2, t]
        prediction_auc[0, t] = i
        prediction_auc[1, t] = j
        prediction_auc[2, t] = prediction[i, j]
        if prediction_01[i, j] == truth == 1.0:
            TP = TP + 1
        elif prediction_01[i, j] == truth == 0.0:
            TN = TN + 1
        elif prediction_01[i, j] != truth and truth == 1.0:
            FN = FN + 1
        elif prediction_01[i, j] != truth and truth == 0.0:
            FP = FP + 1

    # accuracy, precision, recall and F1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 0.00001)
    precision = (TP) / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    F1 = 2 * (precision * recall) / (precision + recall + 0.0001)

    #AUC
    y_truth = []
    for i in range(0, len(groundTruth[0, :])):
        y_truth.append(int(groundTruth[2, i]))

    fpr, tpr, thresholds = metrics.roc_curve(y_truth, prediction_auc[2, :])
    auc = metrics.auc(fpr, tpr)

    return accuracy, recall, precision, auc, F1


def minmaxNoise(X, bound):
    X_normalized = (X - X.min()) / (X.max() - X.min()) * (bound - (-bound)) - bound
    return X_normalized


def minmax(X):
    X_normalized = (X - X.min()) / (X.max() - X.min()) * ((1 - 0) + 0)
    return X_normalized


def predict(U, UBias, ULatentScaler, WBilinear, sideBilinear):

    prediction = U.T @ ULatentScaler @ U
    for i in range(0, len(U[0, :])):
        for j in range(0, len(U[0, :])):
            prediction[i, j] = sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j] + prediction[i, j] + UBias[i] + UBias[j]

    predictionS = 1. / (1 + np.exp(-prediction))  # predicted probability

    return predictionS


def modelGen(D, sideBilinear, weightClass, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Bilinear, epochs, expCounter, alpha, epsilon, testSet):

    #try:
    print("Model: ", expCounter)
    master(D, expCounter, sideBilinear, weightClass, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Bilinear, epochs, alpha, epsilon, testSet)  # obtain metrics
    expCounter += 1
    #except:
       #print("Error for model:", expCounter)


# nodes in the graph
nodes = 130
# number of latent features
latentFeatures = 5


# loading of moniadic features

sideBilinear = np.loadtxt(open("moniadic.csv", "rb"), delimiter=",")
sideBilinear = np.delete(sideBilinear, 0, 1)
sideBilinear = np.transpose(sideBilinear)
sideBilinear = minmax(sideBilinear)


importFile = open("7_file.pkl", "rb")
model = pickle.load(importFile)
importFile.close()

D = model.CVSet

TRAIN_RATIO = 0.9

TestSet = model.testSet

print("Dimension TrainingSet: ", len(D[0, :]))
print("Dimension TestSet: ", len(TestSet[0, :]))

experimentCounter = 0

# 10 latent features for 792 nodes
weights = weight(latentFeatures, nodes)
l4mbda = lambdas()
eta = eta()
loss = 'log'
link = 'sigmoid'
p = mp.Pool(1)
expcounter = 0

for eta_Latent in eta.etaLatent:
    for eta_RowBias in eta.etaRowBias:
        for eta_LatentScaler in eta.etaLatentScaler:
            for eta_Bilinear in eta.etaBilinear:
                for epochs in eta.epochs:
                    p.apply(modelGen, args=(D, sideBilinear, weights, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Bilinear, epochs, expcounter, 1, 0.5, TestSet))
                    expcounter += 1
p.close()
p.join()





