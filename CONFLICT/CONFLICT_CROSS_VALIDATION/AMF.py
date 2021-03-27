import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp

dB = 3
dP = 5
folds = 8
aucStep = 5


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



class weight:
    def __init__(self, p, n):
        self.U = np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[p, n]).astype(np.float128))  # for k latent features and m nodes
        self.UBias = np.random.choice([-0.000000000000000000000000001, 0.000000000000000000000000000001], size=[n]).astype(np.float128)
        self.ULatentScaler = np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[p, p]).astype(np.float128))  # for asymmetric; for symmetric, use diag(randn(k)) instead
        self.WPair =  np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[1, dP]).astype(np.float128)) # for dPair features for each pair
        self.WBilinear = np.asmatrix(np.random.choice([-0.0001, 0.0001], size=[dB, dB]).astype(np.float128)) # for dBilinear features for each node


class lambdas:
    def __init__(self):
        self.lambdaLatent = [0]  # regularization for node's latent vector U
        self.lambdaRowBias = [0] # regularization for node's bias UBias
        self.lambdaLatentScaler = [0]  # regularization for scaling factors Lambda (in paper)
        self.lambdaPair = [0]  # regularization for weights on pair features
        self.lambdaBilinear = [0]  # regularization for weights on node features
        self.lambdaScaler = [0] # scaling factor for regularization, can be set to 1 by default


class eta:
    def __init__(self):
        self.etaLatent = [0.05, 0.01] #[0.01, 0.1, 0.5]  # learning rate for latent feature
        self.etaRowBias = [0.005, 0.0005]#[0.001, 0.01, 0.1]  # learning rate for node bias
        self.etaLatentScaler = [0.05]#[0.01, 0.1, 0.5]  # learning rate for scaler to latent features
        self.etaPair = [0.05]
        self.etaBilinear = [0.05, 0.005]#[0.01, 0.1, 0.5]
        self.etaBias = 0.000005 # learning rate for global bias, used when features are present
        self.epochs = [1000] #passes of SGD


def sgd(D, sidePair, sideBilinear, weights, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Pair, eta_Bilinear, epochs, plot0, plot0HR, validationSet, fold):

    pairs = len(D[1, :])
    testOnes = []
    for index in range(0, len(validationSet[0, :])):
        if validationSet[2, index] == 1.0:
            testOnes.append(index)

    #Load model parameters into sgd
    U = np.copy(weights.U)
    UBias = np.copy(weights.UBias)
    ULatentScaler = np.copy(weights.ULatentScaler)
    WPair = np.copy(weights.WPair)
    WBilinear = np.copy(weights.WBilinear)

    #load initial learning rates
    etaLatent0 = eta_Latent
    etaLatentScaler0 = eta_LatentScaler
    etaPair0 = eta_Pair
    etaBilinear0 = eta_Bilinear
    etaRowBias0 = eta_RowBias

    #set the counter for the plotting (aucCounter is used both to plot AUC and HitRatio)
    aucCounter = 0

    # Main SGD body
    for e in range(1, epochs+1):
        # Dampening of the learning rates across epochs
        etaLatent = etaLatent0  # / ((1 + etaLatent0 * lambdaLatent) * e)
        etaRowBias = etaRowBias0  # / ((1 + etaRowBias0 * lambdaRowBias) * e)
        etaLatentScaler = etaLatentScaler0  # / ((1 + etaLatentScaler0 * lambdaLatentScaler) * e)
        etaPair = etaPair0  # / ((1 + etaPair0 * lambdaPair) * e)
        etaBilinear = etaBilinear0  # / ((1 + etaBilinear0 * lambdaBilinear) * e)

        limit = int(1 * pairs)


        for t in range(0, limit):
            i = int(D[0, t]) - 1
            j = int(D[1, t]) - 1
            truth = int(D[2, t])

            prediction = U[:, i].T @ ULatentScaler @ U[:, j] + UBias[i] + UBias[j] + sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j] #+ WPair @ sidePair[:, i, j]
            predictionAdjust = prediction

            sigma = 1 / (1 + np.exp(-predictionAdjust))  # this must be a matrix of link probabilities.

            #cost = -(truth*np.log(sigma)+(1-truth)*(np.log(1-sigma)))

            #gradients computation
            gradscaler = float(sigma - truth)
            gradI = ULatentScaler @ U[:, j]
            gradJ = (U[:, i].T @ ULatentScaler).T
            gradRowBias = 1
            gradPair = sidePair[:, i, j].T
            gradLatentScaler = U[:, i] @ U[:, j].T
            gradBilinear = sideBilinear[:, i] @ sideBilinear[:, j].T

            #update section
            U[:, i] = U[:, i] - etaLatent * (gradscaler * gradI)  # + lambdaLatent * lambdaScaler * U[:, i])
            U[:, j] = U[:, j] - etaLatent * (gradscaler * gradJ)  # + lambdaLatent * lambdaScaler * U[:, j])
            UBias[i] = UBias[i] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[i])
            UBias[j] = UBias[j] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[j])
            #WPair = WPair - etaPair * (gradscaler * gradPair)  # + lambdaPair * WPair)
            ULatentScaler = ULatentScaler - etaLatentScaler * (gradscaler * gradLatentScaler)  # + lambdaLatentScaler * ULatentScaler)
            WBilinear = WBilinear - etaBilinear * (gradscaler * gradBilinear)  # + lambdaBilinear * WBilinear)

        #metrics computation
        if e % aucStep == 0:
            prediction = predict(U, UBias, ULatentScaler, WPair, WBilinear, sideBilinear, sidePair)
            acc, rec, prec, auc, f1 = test(prediction, validationSet)
            hr = hitratio(U, ULatentScaler, UBias, WPair, WBilinear, validationSet, testOnes)
            plot0[aucCounter, fold] = auc
            plot0HR[aucCounter, fold] = hr
            aucCounter += 1

    return U, UBias, ULatentScaler, WPair, WBilinear, plot0, plot0HR, aucCounter


def sgd_adv(D, sidePair, sideBilinear, U0, UBias0, ULatentScaler0, WPair0, WBilinear0, eta_Latent, eta_LatentScaler, eta_Pair, eta_Bilinear, eta_RowBias, epochs, plotAdv, plotAdvHR, validationSet, fold):

    pairs = len(D[1, :])
    testOnes = []

    for index in range(0, len(validationSet[0, :])):
        if validationSet[2, index] == 1.0:
            testOnes.append(index)

    #load model parameter, passed from the initial training SGD function
    U = np.copy(U0)
    UBias = np.copy(UBias0)
    ULatentScaler = np.copy(ULatentScaler0)
    WPair = np.copy(WPair0)
    WBilinear = np.copy(WBilinear0)

    #load starting learning rates
    etaLatent0 = eta_Latent
    etaLatentScaler0 = eta_LatentScaler
    etaPair0 = eta_Pair
    etaBilinear0 = eta_Bilinear
    etaRowBias0 = float(eta_RowBias)

    #load noise parameters
    alpha = 1
    epsilon = 0.5

    aucCounter = 0

    for e in range(1, epochs):
        # Dampening of the learning rates across epochs
        etaLatent = etaLatent0 #/ ((1 + etaLatent0 * lambdaLatent) * e)
        etaRowBias = etaRowBias0 #/ ((1 + etaRowBias0 * lambdaRowBias) * e)
        etaLatentScaler = etaLatentScaler0 #/ ((1 + etaLatentScaler0 * lambdaLatentScaler) * e)
        etaPair = etaPair0 #/ ((1 + etaPair0 * lambdaPair) * e)
        etaBilinear = etaBilinear0 #/ ((1 + etaBilinear0 * lambdaBilinear) * e)

        limit = int(1 * pairs)

        for t in range(0, limit):
            i = int(D[0, t]) - 1
            j = int(D[1, t]) - 1
            truth = int(D[2, t])

            #here we set the bound for the random noise that could be 0 if we want to add Delta=0 or the commented value if we want ||Delta||<epsilon
            noiseBound = 0 # np.sqrt(np.power(epsilon, 2)/len(U[:, 0]))

            # Procedure for the adversarial perturbation buidling

            #DeltaI/J are relative to Ui and Uj
            DeltaI = np.random.rand(len(U[:, 0]))
            DeltaJ = np.random.rand(len(U[:, 0]))

            #DeltaXI/J are relative to Xi and Xj
            DeltaXI = np.random.rand(len(sideBilinear[:, 0]))
            DeltaXJ = np.random.rand(len(sideBilinear[:, 0]))

            DeltaI = minmaxNoise(DeltaI, noiseBound)
            DeltaJ = minmaxNoise(DeltaJ, noiseBound)
            DeltaXI = minmaxNoise(DeltaXI, noiseBound)
            DeltaXJ = minmaxNoise(DeltaXJ, noiseBound)

            predictionDelta = (U[:, i].T+DeltaI) @ ULatentScaler @ (U[:, j].T + DeltaJ).T + UBias[i] + UBias[j] \
                              + (sideBilinear[:, i].T + DeltaXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaXJ).T #WPair @ sidePair[:, i, j]  # + WBias

            sigmaDelta = 1 / (1 + np.exp(-predictionDelta))  # this must be a matrix of link probabilities.

            GammaI = alpha * (sigmaDelta - truth) * (ULatentScaler @ (U[:, j].T+DeltaJ).T).T
            GammaJ = alpha * (sigmaDelta - truth) * ((U[:, i].T+DeltaI) @ ULatentScaler)
            GammaXI = alpha * (sigmaDelta - truth) * (WBilinear @ (sideBilinear[:, j].T+DeltaXJ).T).T
            GammaXJ = alpha * (sigmaDelta - truth) * ((sideBilinear[:, i].T+DeltaXI) @ WBilinear)

            DeltaAdvI = epsilon * GammaI / np.sqrt(max(np.sum(np.power(GammaI, 2)), 0.000001))
            DeltaAdvJ = epsilon * GammaJ / np.sqrt(max(np.sum(np.power(GammaJ, 2)), 0.000001))
            DeltaAdvXI = epsilon * GammaXI / np.sqrt(max(np.sum(np.power(GammaXI, 2)), 0.000001))
            DeltaAdvXJ = epsilon * GammaXJ / np.sqrt(max(np.sum(np.power(GammaXJ, 2)), 0.000001))

            predictionAdv = (U[:, i].T + DeltaAdvI) @ ULatentScaler @ (U[:, j].T + DeltaAdvJ).T + UBias[i] + UBias[j] \
                            + (sideBilinear[:, i].T + DeltaAdvXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaAdvXJ).T #+ WPair @ sidePair[:, i, j]
            prediction = (U[:, i]).T @ ULatentScaler @ U[:, j] + UBias[i] + UBias[j] + sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j]

            sigmaAdv = 1 / (1 + np.exp(-predictionAdv))
            sigma = 1 / (1 + np.exp(-prediction))

            #cost = -(truth * np.log(sigma) + (1 - truth) * (np.log(1-sigma))) - alpha * ((truth * np.log(sigmaAdv)) + (1 - truth) * (np.log(1-sigmaAdv)))

            gradscalerAdv = float(sigmaAdv - truth)
            gradscaler = float(sigma - truth)

            #gradients computation
            gradIAdv = ULatentScaler @ (U[:, j].T + DeltaAdvJ).T
            gradI = ULatentScaler @ U[:, j]
            gradJAdv = ((U[:, i].T + DeltaAdvI) @ ULatentScaler).T
            gradJ = (U[:, i].T @ ULatentScaler).T
            gradPair = sidePair[:, i, j].T
            gradBilinear = sideBilinear[:, i] @ sideBilinear[:, j].T
            gradBilinearAdv = sideBilinear[:, i] @ sideBilinear[:, j].T + sideBilinear[:, j] @ DeltaAdvXI + sideBilinear[:, i] @ DeltaAdvXJ + DeltaAdvXI @ DeltaAdvXJ.T
            gradBias = 1
            gradLatentScaler = U[:, i] @ U[:, j].T
            gradLatentScalerAdv = U[:, i] @ U[:, j].T + U[:, j] @ DeltaAdvI + U[:, i] @ DeltaAdvJ + DeltaAdvI @ DeltaAdvJ.T

            #updates
            U[:, i] = U[:, i] - etaLatent * (gradscaler * gradI + alpha * gradscalerAdv * gradIAdv)
            U[:, j] = U[:, j] - etaLatent * (gradscaler * gradJ + alpha * gradscalerAdv * gradJAdv)
            UBias[i] = UBias[i] - etaRowBias * (gradscaler * gradBias)
            UBias[j] = UBias[j] - etaRowBias * (gradscaler * gradBias)
            #WPair = WPair - etaPair * (gradscaler * gradPair + alpha * gradscalerAdv * gradBilinearAdv)
            ULatentScaler = ULatentScaler - etaLatentScaler * (gradscaler * gradLatentScaler + alpha * gradscalerAdv * gradLatentScalerAdv)
            WBilinear = WBilinear - etaBilinear * (gradscaler * gradBilinear + alpha * gradscalerAdv * gradBilinearAdv)

        if e % aucStep == 0:
            prediction = predict(U, UBias, ULatentScaler, WPair, WBilinear, sideBilinear, sP)
            acc, rec, prec, auc, f1 = test(prediction, validationSet)
            hr = hitratio(U, ULatentScaler, UBias, WPair, WBilinear, validationSet, testOnes)
            plotAdvHR[aucCounter, fold] = hr
            plotAdv[aucCounter, fold] = auc
            aucCounter += 1

    return U, UBias, ULatentScaler, WPair, WBilinear, plotAdv, plotAdvHR


def sgd_continue(D, sidePair, sideBilinear, U, UBiasContinue, ULatentScaler, WPair, WBilinear, eta_Latent, eta_LatentScaler, eta_Pair, eta_Bilinear, eta_RowBias, epochs, plot0, plot0HR, validationSet, fold, plotPoint):

    pairs = len(D[1, :])
    testOnes = []

    for index in range(0, len(validationSet[0, :])):
        if validationSet[2, index] == 1.0:
            testOnes.append(index)

    U0 = np.copy(U)
    U0Bias = np.copy(UBiasContinue)
    ULatentScaler = np.copy(ULatentScaler)
    WPair = np.copy(WPair)
    WBilinear = np.copy(WBilinear)


    etaLatent0 = eta_Latent
    etaLatentScaler0 = eta_LatentScaler
    etaPair0 = eta_Pair
    etaBilinear0 = eta_Bilinear
    etaRowBias0 = eta_RowBias

    U = U0
    UBias = U0Bias

    aucCounter = plotPoint

    # Main SGD body
    for e in range(1, epochs):
        # Dampening of the learning rates across epochs
        etaLatent = etaLatent0 #/ ((1 + etaLatent0 * lambdaLatent) * e)
        etaRowBias = etaRowBias0 #/ ((1 + etaRowBias0 * lambdaRowBias) * e)
        etaLatentScaler = etaLatentScaler0 #/ ((1 + etaLatentScaler0 * lambdaLatentScaler) * e)
        etaPair = etaPair0 #/ ((1 + etaPair0 * lambdaPair) * e)
        etaBilinear = etaBilinear0 #/ ((1 + etaBilinear0 * lambdaBilinear) * e)

        limit = int(1 * pairs)

        for t in range(0, limit):

            i = int(D[0, t]) - 1
            j = int(D[1, t]) - 1
            truth = int(D[2, t])

            prediction = U[:, i].T @ ULatentScaler @ U[:, j] + UBias[i] + UBias[j] + sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j]# + WPair @ sidePair[:, i, j] #+ WBias

            sigma = 1 / (1 + np.exp(-prediction))

            cost = -(truth*np.log(sigma)+(1-truth)*(np.log(1-sigma)))

            #Common gradient scaler
            gradscaler = float(sigma - truth)

            gradI = ULatentScaler @ U[:, j]
            gradJ = (U[:, i].T @ ULatentScaler).T
            gradRowBias = 1  # ones(1, numel(examples))        # 1 x 1
            gradPair = sidePair[:, i, j].T  # diventa 1x5
            gradLatentScaler = U[:, i] @ U[:, j].T
            gradBilinear = sideBilinear[:, i] @ sideBilinear[:, j].T

            U[:, i] = U[:, i] - etaLatent * (gradscaler * gradI)  # + lambdaLatent * lambdaScaler * U[:, i])  # U_i è di dimensione 5x1
            U[:, j] = U[:, j] - etaLatent * (gradscaler * gradJ)  # + lambdaLatent * lambdaScaler * U[:, j])
            UBias[i] = UBias[i] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[i])
            UBias[j] = UBias[j] - etaRowBias * (gradscaler * gradRowBias)  # + lambdaRowBias * UBias[j])
            #WPair = WPair - etaPair * (gradscaler * gradPair)  # + lambdaPair * WPair)
            ULatentScaler = ULatentScaler - etaLatentScaler * (gradscaler * gradLatentScaler)  # + lambdaLatentScaler * ULatentScaler)
            WBilinear = WBilinear - etaBilinear * (gradscaler * gradBilinear)  # + lambdaBilinear * WBilinear)

        if e % aucStep == 0:
            prediction = predict(U, UBias, ULatentScaler, WPair, WBilinear, sideBilinear, sP)
            acc, rec, prec, auc, fot1 = test(prediction, validationSet)
            plot0[aucCounter, fold] = auc
            hr = hitratio(U, ULatentScaler, UBias, WPair, WBilinear, validationSet, testOnes)
            plot0HR[aucCounter, fold] = hr
            aucCounter += 1

    return U, UBias, ULatentScaler, WPair, WBilinear, plot0, plot0HR


def hitratio(U, ULatentScaler, UBias, WPair, WBilinear, test, testOnes):

    #test: [3xn]: row1=nodo1 |row2=nodo2 | row3=0/1
    #testOnes: vector containing indexer about positive test samples (samples with raw3=1)

    divider = len(testOnes)
    hr = 0
    predictionNode = np.random.rand(len(U[0, :]), 2) #create a 130x2 table
    indexes = np.arange(1, (len(U[0, :]) + 1))
    predictionNode[:, 1] = np.copy(indexes) #in the second column inject a vector of indexes from 1 to 130

    limit = len(U[0, :])
    for node in range(0, limit):
        vectorPred = U[:, node].T @ ULatentScaler @ U + UBias[node] + UBias + sideBilinear[:, node] @ WBilinear @ sideBilinear #[1x10]x[10x10]x[10x792] + [1] + [1x792] + [1x209]x[209x209]x[209x792] = [1x792]
        predictionNode[:, 0] = np.copy(vectorPred) #insert the prediction in the table
        predictionNode = predictionNode[np.argsort(predictionNode[:, 0])[::-1]] #sort the table in descending order based on the first column

        #Iterate on the first N elements of the prediction table. In amongh the first N is present an element with positive edge, then increse "hr"
        for ii in range(0, 10): #HR@10
            for jj in testOnes:
                if int(test[0, jj]) == node and test[1, jj] == predictionNode[ii, 1]:
                    hr += 1

        #Once the hr summation is completed and all the "hits" have been discovered, normalize with respect to the number positive samples in the test set
    hr = hr / divider
    return hr


def kfold(DCv, folds,  expCounter, sP, sideBilinear, weightClass, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Pair, eta_Bilinear, epochs):

    # DCv is the Cross Validation Set: The entire graph without 10% of edges
    if folds != 0:
        step = 1/folds
    else:
        print("error")
    v_cross_validation_set = []  # set of indexes of the DCv where exists and edge

    for index in range(0, len(DCv[0, :])):
        if DCv[2, index] == 1.0:
            v_cross_validation_set.append(index)
    setLength = len(DCv[0, :])

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

    #newepochs is the number of epochs for the sgd_adv learning
    if epochs % 2 == 0:
        Newepochs = int(epochs / 2)
    else:
        Newepochs = int((epochs + 1) / 2)

    #plot parameters
    aucSamples = int(math.ceil((epochs+Newepochs)/aucStep))
    aucSamplesAdv = int(math.ceil(Newepochs/aucStep))
    plot0 = np.random.rand(aucSamples-1, folds)
    plotAdv = np.random.rand(aucSamplesAdv-1, folds)
    plot0final = np.random.rand(aucSamples-1)
    plotAdvfinal = np.random.rand(aucSamplesAdv-1)

    plot0HR = np.random.rand(aucSamples-1, folds)
    plotAdvHR = np.random.rand(aucSamplesAdv-1, folds)
    plot0finalHR = np.random.rand(aucSamples-1)
    plotAdvfinalHR = np.random.rand(aucSamplesAdv-1)

    for fold in range(0, folds):

        tSet = np.copy(DCv[:, :])
        trainSet = np.delete(tSet, np.s_[math.ceil(fold * step * setLength): math.ceil(((fold+1) * step * setLength)-1)], axis=1)

        validationSet = DCv[:, math.ceil(fold * step * setLength):math.ceil(((fold+1) * step * setLength)-1)]

        #print to keep track of the progress
        print("fold %d of model %d" % (fold, expCounter))

        #initial training and prediction
        U, UBias, ULatentScaler, WPair, WBilinear, plot0, plot0HR, plotPoint = sgd(trainSet, sP, sideBilinear, weightClass, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Pair, eta_Bilinear, epochs, plot0, plot0HR, validationSet, fold)
        prediction = predict(U, UBias, ULatentScaler, WPair, WBilinear, sideBilinear,  sP)  # generate prediction matrix

        # acc, rec, prec, auc, f1 = test(prediction, validationSet)  # obtain metrics

        # adversarial training
        UAdv, UBiasAdv, ULatentScalerAdv, WPairAdv, WBilinearAdv, plotAdv, plotAdvHR = sgd_adv(trainSet, sP, sideBilinear, U, UBias, ULatentScaler, WPair, WBilinear, eta_Latent, eta_LatentScaler, eta_Pair, eta_Bilinear, eta_RowBias, Newepochs, plotAdv, plotAdvHR, validationSet, fold)
        # continue the initial training
        UCon, UBiasCon, ULatentScalerCon, WPairCon, WBilinearCon, plot0, plot0HR = sgd_continue(trainSet, sP, sideBilinear, U, UBias, ULatentScaler, WPair, WBilinear, eta_Latent,
                    eta_LatentScaler, eta_Pair, eta_Bilinear, eta_RowBias, Newepochs, plot0, plot0HR, validationSet, fold, plotPoint)

        predictionAdv = predict(UAdv, UBiasAdv, ULatentScalerAdv, WPairAdv, WBilinearAdv, sideBilinear, sP)
        predictionCon = predict(UCon, UBiasCon, ULatentScalerCon, WPairCon, WBilinearCon, sideBilinear, sP)

        accAdv, recAdv, precAdv, aucAdv, f1Adv = test(predictionAdv, validationSet)
        accCon, recCon, precCon, aucCon, f1Con = test(predictionCon, validationSet)

        validationOnes = []

        for index in range(0, len(validationSet[0, :])):
            if validationSet[2, index] == 1.0:
                validationOnes.append(index)

        hrAdv = hitratio(UAdv, ULatentScalerAdv, UBiasAdv, WPairAdv, WBilinearAdv, validationSet, validationOnes)
        hrCon = hitratio(UCon, ULatentScalerCon, UBiasCon, WPairCon, WBilinearCon, validationSet, validationOnes)

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
    plotString = str(expCounter) + '_AUC_plot.jpg'
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

    #Plot the HR
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


    accMean = np.mean(accuracy)
    accStd = np.std(accuracy)
    precMean = np.mean(precision)
    precStd = np.std(precision)
    recMean = np.mean(recall)
    recStd = np.std(recall)
    aucMean = np.mean(AUC)
    aucStd = np.std(AUC)
    f1Mean = np.mean(F1)
    f1Std = np.std(F1)
    hrMean = np.mean(HR)
    hrStd = np.std(HR)

    accMeanAdv = np.mean(accuracyAdv)
    accStdAdv = np.std(accuracyAdv)
    precMeanAdv = np.mean(precisionAdv)
    precStdAdv = np.std(precisionAdv)
    recMeanAdv = np.mean(recallAdv)
    recStdAdv = np.std(recallAdv)
    aucMeanAdv = np.mean(AUCAdv)
    aucStdAdv = np.std(AUCAdv)
    f1MeanAdv = np.mean(F1Adv)
    f1StdAdv = np.std(F1Adv)
    hrMeanAdv = np.mean(HRAdv)
    hrStdAdv = np.std(HRAdv)

    print("MODEL: ", expCounter)
    print("eta_Latent: ", eta_Latent)
    print("eta_RowBias: ", eta_RowBias)
    print("eta_LatentScaler: ", eta_LatentScaler)
    print("eta_Pair: ", eta_Pair)
    print("eta_Bilinear: ", eta_Bilinear)
    print("Epochs per Kfold:", (epochs - 1))
    print("-----------------------------------")
    print("     METRICS for MODEL: ", expCounter)
    print("--- Matrix Factorization ----")
    print("Accuracy: %1.3f +/- %1.3f" % (accMean, accStd))
    print("Precision: %1.3f +/- %1.3f" % (precMean, precStd))
    print("Recall: %1.3f +/- %1.3f" % (recMean, recStd))
    print("F1: %1.3f +/- %1.3f" % (f1Mean, f1Std))
    print("AUC: %1.17f +/- %1.3f" % (aucMean, aucStd))
    print("HR@10: %1.17f +/- %1.3f" % (hrMean, hrStd))
    print("---- Adversarial Matrix Factorization ----")
    print("Accuracy: %1.3f +/- %1.3f" % (accMeanAdv, accStdAdv))
    print("Precision: %1.3f +/- %1.3f" % (precMeanAdv, precStdAdv))
    print("Recall: %1.3f +/- %1.3f" % (recMeanAdv, recStdAdv))
    print("F1: %1.3f +/- %1.3f" % (f1MeanAdv, f1StdAdv))
    print("AUC: %1.17f +/- %1.3f" % (aucMeanAdv, aucStdAdv))
    print("HR@10: %1.17f +/- %1.3f" % (hrMeanAdv, hrStdAdv))

    #export file
    expString = './exportFiles/' + str(expCounter) + '_file.pkl'
    exportObject = export(eta_Latent, eta_RowBias, eta_LatentScaler, eta_Pair, eta_Bilinear, folds, (epochs - 1), DCv, TestSet)
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


def predict(U, UBias, ULatentScaler, WPair, WBilinear, sideBilinear, sP):
    prediction = U.T @ ULatentScaler @ U

    for i in range(0, len(U[0, :])):
        for j in range(0, len(U[0, :])):
            prediction[i, j] =  sideBilinear[:, i].T @ WBilinear @ sideBilinear[:, j] + prediction[i, j] + UBias[i] + UBias[j] # +  WPair @ sP[:, i, j] #

    predictionS = 1. / (1 + np.exp(-prediction))  # predicted probability

    return predictionS


def modelGen(folds, DCv, sP, sideBilinear, weightClass, lambdaLatent, lambdaRowBias, lambdaLatentScaler, lambdaPair, lambdaBilinear, lambdaScaler,  eta_Latent, eta_RowBias, eta_LatentScaler,
             eta_Pair, eta_Bilinear, epochs,  expCounter):
    try:
        print("Model: ", expCounter)
        kfold(DCv, folds, expCounter, sP, sideBilinear, weightClass, eta_Latent, eta_RowBias, eta_LatentScaler, eta_Pair, eta_Bilinear, epochs)
        expCounter += 1

    except:
        print("Error for model:", expCounter)


# nodes in the graph
m = 130
# number of latent features
k = 5

# loading of moniadic features
sideBilinear = np.loadtxt(open("moniadic.csv", "rb"), delimiter=",")
sideBilinear = np.delete(sideBilinear, 0, 1)
sideBilinear = np.transpose(sideBilinear)
sideBilinear = minmax(sideBilinear)

# loading of dyadic features
sidePair = np.loadtxt(open("dyadic.csv", "rb"), delimiter=",")  # np.random.rand(dP, m + 1, m + 1)
sidePair = np.delete(sidePair, 0, 1)
sidePair = np.delete(sidePair, 0, 1)
sP = np.random.rand(dP, m, m)
t = 0
for i in range(0, 130):
    for j in range(0, 130):
        sP[:, i, j] = sidePair[t]
        t += 1
sP = minmax(sP)

#load the dataset
D = np.transpose(np.loadtxt(open("adjancence.csv", "rb"), delimiter=","))
DShuffle = np.transpose(np.loadtxt(open("adjancence.csv", "rb"), delimiter=","))
permVec = np.random.permutation(range(0, len(D[0, :])))
j = 0
for i in permVec:
    DShuffle[:, j] = D[:, i]
    j = j + 1

TRAIN_RATIO = 0.9

v = []
for i in range(0, len(DShuffle[0, :])):
    if DShuffle[2, i] == 1.0:
        v.append(i)

v = v[0:math.ceil(TRAIN_RATIO*len(v))]
splitPoint = np.max(v)
DCv = np.random.rand(len(DShuffle[:, 0]), splitPoint)
TestSet = np.random.rand(len(DShuffle[:, 0]), (len(DShuffle[0, :])-(splitPoint+1)))

for i in range(0, splitPoint):
    DCv[:, i] = DShuffle[:, i]

testIndex = 0
for i in range(splitPoint+1, len(DShuffle[0, :])):
    TestSet[:, testIndex] = DShuffle[:, i]
    testIndex = testIndex + 1

print("Dimension CVSet: ", len(DCv[0, :]))
print("Dimension TestSet: ", len(TestSet[0, :]))


experimentCounter = 0
weights = weight(5, 130)
l4mbda = lambdas()
eta = eta()
loss = 'log'
link = 'sigmoid'

p = mp.Pool(3)
expcounter = 0

for folds in [folds]:
    for lambda_Latent in l4mbda.lambdaLatent:
        for lambda_RowBias in l4mbda.lambdaRowBias:
            for lambda_LatentScaler in l4mbda.lambdaLatentScaler:
                for lambda_Pair in l4mbda.lambdaPair:
                    for lambda_Bilinear in l4mbda.lambdaBilinear:
                        for lambda_Scaler in l4mbda.lambdaScaler:
                            for eta_Latent in eta.etaLatent:
                                for eta_RowBias in eta.etaRowBias:
                                    for eta_LatentScaler in eta.etaLatentScaler:
                                        for eta_Pair in eta.etaPair:
                                            for eta_Bilinear in eta.etaBilinear:
                                                for epochs in eta.epochs:

                                                    p.apply_async(modelGen, args=(folds, DCv, sP, sideBilinear, weights, lambda_Latent, lambda_RowBias, lambda_LatentScaler, lambda_Pair, lambda_Bilinear, lambda_Scaler,  eta_Latent, eta_RowBias, eta_LatentScaler, eta_Pair, eta_Bilinear, epochs,  expcounter))
                                                    expcounter += 1
p.close()
p.join()





