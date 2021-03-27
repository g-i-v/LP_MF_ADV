import numpy as np
from sklearn import metrics, decomposition
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


def hitratioRandomNoise(U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, sideBilinear, test, testOnes, noiseBound):

    #test: [3xn]: row1=nodo1 |row2=nodo2 | row3=0/1
    #testOnes: vector containing indexer about positive test samples (samples with raw3=1)

    divider = len(testOnes)
    hr = 0
    hrAdv = 0
    predictionNode = np.random.rand(len(U[0, :]), 2)
    predictionNodeAdv = np.random.rand(len(U[0, :]), 2)

    indexes = np.arange(1, (len(U[0, :]) + 1))

    predictionNode[:, 1] = np.copy(indexes)
    predictionNodeAdv[:, 1] = np.copy(indexes)
    limit = len(U[0, :])

    DeltaU = np.random.rand(len(U[:, 0]), len(U[0, :]))
    DeltaX = np.random.rand(len(sideBilinear[:, 0]), len(sideBilinear[0, :]))
    DeltaU = minmaxNoise(DeltaU, noiseBound)
    DeltaX = minmaxNoise(DeltaX, noiseBound)

    UAdv = UAdv + DeltaU
    U = U+DeltaU
    sB = sideBilinear+DeltaX

    for node in range(0, limit):
        prediction = U[:, node].T @ ULatentScaler @ U + UBias[node] + UBias + sB[:, node] @ WBilinear @ sB
        predictionNode[:, 0] = np.copy(prediction)
        predictionNode = predictionNode[np.argsort(predictionNode[:, 0])[::-1]]
        predictionAdv = UAdv[:, node].T @ ULatentScalerAdv @ U + UBiasAdv[node] + UBiasAdv + sB[:, node] @ WBilinearAdv @ sB
        predictionNodeAdv[:, 0] = np.copy(predictionAdv)
        predictionNodeAdv = predictionNodeAdv[np.argsort(predictionNodeAdv[:, 0])[::-1]]

        for ii in range(0, 10):
            for jj in testOnes:
                if int(test[0, jj]) == node and test[1, jj] == predictionNode[ii, 1]:
                    hr += 1
                elif int(test[0, jj]) == node and test[1, jj] == predictionNodeAdv[ii, 1]:
                    hrAdv += 1
    hr = hr / divider
    hrAdv = hrAdv / divider
    return hrAdv, hr


def hitratioAdversarialNoise(D , U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv,
                        sideBilinear, test, testOnes, noiseBound):
    # test: [3xn]: row1=nodo1 |row2=nodo2 | row3=0/1
    # testOnes: vector containing indexer about positive test samples (samples with raw3=1)



    alpha = 1
    epsilon = 0.5

    divider = len(testOnes)
    hr = 0
    hrAdv = 0
    predictionNode = np.random.rand(len(U[0, :]), 2)
    predictionNodeAdv = np.random.rand(len(U[0, :]), 2)

    indexes = np.arange(1, (len(U[0, :]) + 1))

    predictionNode[:, 1] = np.copy(indexes)
    predictionNodeAdv[:, 1] = np.copy(indexes)
    limit = len(U[0, :])

    t = 0
    Ds = np.random.rand(limit, limit)
    for r in range(0, limit):
        for c in range(0, limit):
            Ds[r, c] = D[2, t]
            t += 1

    predictionAdv = np.zeros(limit)
    prediction = np.zeros(limit)
    for node in range(0, limit):
        for destination in range(0, limit):

            truth = Ds[node, destination]

            DeltaI = np.zeros(len(U[:, 0]))
            DeltaJ = np.zeros(len(U[:, 0]))
            DeltaXI = np.zeros(len(sideBilinear[:, 0]))
            DeltaXJ = np.zeros(len(sideBilinear[:, 0]))

            predAdv = (UAdv[:, node].T + DeltaI) @ ULatentScaler @ (UAdv[:, destination].T + DeltaJ).T + UBiasAdv[node]\
                      + UBiasAdv[destination] + (sideBilinear[:, node].T + DeltaXI) @ WBilinearAdv @ (sideBilinear[:, destination].T + DeltaXJ).T
            pred = (U[:, node].T + DeltaI) @ ULatentScaler @ (U[:, destination].T + DeltaJ).T + UBias[node] + UBias[destination] + (
                    sideBilinear[:, node].T + DeltaXI) @ WBilinear @ (sideBilinear[:, destination].T + DeltaXJ).T

            sigmaDeltaAdv = 1 / (1 + np.exp(-predAdv))
            sigmaDelta = 1 / (1 + np.exp(-pred))

            # Gamma and DeltaAdv generation for AMF
            GammaIAdv = alpha * (sigmaDeltaAdv - truth) * (ULatentScalerAdv @ (UAdv[:, destination].T + DeltaJ).T).T
            GammaJAdv = alpha * (sigmaDeltaAdv - truth) * ((UAdv[:, node].T + DeltaI) @ ULatentScalerAdv)
            GammaXIAdv = alpha * (sigmaDeltaAdv - truth) * (WBilinearAdv @ (sideBilinear[:, destination].T + DeltaXJ).T).T
            GammaXJAdv = alpha * (sigmaDeltaAdv - truth) * ((sideBilinear[:, node].T + DeltaXI) @ WBilinearAdv)
            DeltaAdvIAdv = epsilon * GammaIAdv / np.sqrt(max(np.sum(np.power(GammaIAdv, 2)), 0.000001))
            DeltaAdvJAdv = epsilon * GammaJAdv / np.sqrt(max(np.sum(np.power(GammaJAdv, 2)), 0.000001))
            DeltaAdvXIAdv = epsilon * GammaXIAdv / np.sqrt(max(np.sum(np.power(GammaXIAdv, 2)), 0.000001))
            DeltaAdvXJAdv = epsilon * GammaXJAdv / np.sqrt(max(np.sum(np.power(GammaXJAdv, 2)), 0.000001))

            # Gamma and DeltaAdv generation for MF
            GammaI = alpha * (sigmaDelta - truth) * (ULatentScaler @ (U[:, destination].T + DeltaJ).T).T
            GammaJ = alpha * (sigmaDelta - truth) * ((U[:, node].T + DeltaI) @ ULatentScaler)
            GammaXI = alpha * (sigmaDelta - truth) * (WBilinear @ (sideBilinear[:, destination].T + DeltaXJ).T).T
            GammaXJ = alpha * (sigmaDelta - truth) * ((sideBilinear[:, node].T + DeltaXI) @ WBilinear)
            DeltaAdvI = epsilon * GammaI / np.sqrt(max(np.sum(np.power(GammaI, 2)), 0.000001))
            DeltaAdvJ = epsilon * GammaJ / np.sqrt(max(np.sum(np.power(GammaJ, 2)), 0.000001))
            DeltaAdvXI = epsilon * GammaXI / np.sqrt(max(np.sum(np.power(GammaXI, 2)), 0.000001))
            DeltaAdvXJ = epsilon * GammaXJ / np.sqrt(max(np.sum(np.power(GammaXJ, 2)), 0.000001))

            prediction[destination] = (U[:, node].T+DeltaAdvI) @ ULatentScaler @ (U[:, destination].T + DeltaAdvJ).T + UBias[node] + UBias[destination] + (sideBilinear[:, node].T + DeltaAdvXI) @ WBilinear @ (sideBilinear[:, destination].T + DeltaAdvXJ).T
            predictionNode[:, 0] = np.copy(prediction)
            predictionNode = predictionNode[np.argsort(predictionNode[:, 0])[::-1]]
            predictionAdv[destination] = (UAdv[:, node].T+DeltaAdvIAdv) @ ULatentScalerAdv @ (UAdv[:, destination].T + DeltaAdvJAdv).T + UBiasAdv[node] + UBiasAdv[destination] + (sideBilinear[:, node].T + DeltaAdvXIAdv) @ WBilinearAdv @ (sideBilinear[:, destination].T + DeltaAdvXJAdv).T
            predictionNodeAdv[:, 0] = np.copy(predictionAdv)
            predictionNodeAdv = predictionNodeAdv[np.argsort(predictionNodeAdv[:, 0])[::-1]]

        for ii in range(0, 10):
            for jj in testOnes:
                if int(test[0, jj]) == node and test[1, jj] == predictionNode[ii, 1]:
                    hr += 1
                elif int(test[0, jj]) == node and test[1, jj] == predictionNodeAdv[ii, 1]:
                    hrAdv += 1
    hr = hr / divider
    hrAdv = hrAdv / divider
    return hrAdv, hr


def master(D, U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, sideBilinear, testSet):

    epsilon = 0.5
    alpha = 1

    testOnes = []
    for index in range(0, len(testSet[0, :])):
        if testSet[2, index] == 1.0:
            testOnes.append(index)

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

    noiseBound = np.sqrt(np.power(epsilon, 2)/len(U[0, :]))
    limit = len(U[0, :])

    # Prediction WITHOUT NOISE
    predictionAdv = predict(UAdv, UBiasAdv, ULatentScalerAdv, WBilinearAdv, sideBilinear)  # generate prediction matrix
    prediction = predict(U, UBias, ULatentScaler, WBilinear, sideBilinear)  # generate prediction matrix

    for randomAttacks in range(0, 1):
        # Prediction WITH RANDOM NOISE
        predictionDeltaAdv = np.random.rand(limit, limit)
        predictionDelta = np.random.rand(limit, limit)
        for i in range(0, limit):
            for j in range(0, limit):

                DeltaI = np.random.rand(len(U[:, 0]))
                DeltaJ = np.random.rand(len(U[:, 0]))
                DeltaXI = np.random.rand(len(sideBilinear[:, 0]))
                DeltaXJ = np.random.rand(len(sideBilinear[:, 0]))

                DeltaI = minmaxNoise(DeltaI, noiseBound)
                DeltaJ = minmaxNoise(DeltaJ, noiseBound)
                DeltaXI = minmaxNoise(DeltaXI, noiseBound)
                DeltaXJ = minmaxNoise(DeltaXJ, noiseBound)

                predictionDeltaAdv[i, j] = (UAdv[:, i].T + DeltaI) @ ULatentScaler @ (UAdv[:, j].T + DeltaJ).T + UBiasAdv[i] + UBiasAdv[j] + (
                            sideBilinear[:, i].T + DeltaXI) @ WBilinearAdv @ (sideBilinear[:, j].T + DeltaXJ).T

                predictionDelta[i, j] = (U[:, i].T + DeltaI) @ ULatentScaler @ (U[:, j].T + DeltaJ).T + UBias[i] + UBias[j] + (
                            sideBilinear[:, i].T + DeltaXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaXJ).T

        predictionRandomNoiseAdv = 1./(1+np.exp(-predictionDeltaAdv))
        predictionRandomNoise = 1. / (1 + np.exp(-predictionDelta))

        # Metrics WITH RANDOM NOISE
        accRandomNoiseAdv, recRandomNoiseAdv, precRandomNoiseAdv, aucRandomNoiseAdv, f1RandomNoiseAdv = test(predictionRandomNoiseAdv, testSet)
        accRandomNoise, recRandomNoise, precRandomNoise, aucRandomNoise, f1RandomNoise = test(predictionRandomNoise, testSet)

        # HR WITH RANDOM NOISE
        hrRandomNoiseAdv, hrRandomNoise = hitratioRandomNoise(U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv,
                                                              UBiasAdv, WBilinearAdv, sideBilinear, testSet, testOnes,
                                                              noiseBound)
        accuracy.append(accRandomNoise)
        precision.append(recRandomNoise)
        recall.append(recRandomNoise)
        AUC.append(aucRandomNoise)
        F1.append(f1RandomNoise)
        HR.append(hrRandomNoise)

        accuracyAdv.append(accRandomNoiseAdv)
        precisionAdv.append(recRandomNoiseAdv)
        recallAdv.append(recRandomNoiseAdv)
        AUCAdv.append(aucRandomNoiseAdv)
        F1Adv.append(f1RandomNoiseAdv)
        HRAdv.append(hrRandomNoiseAdv)

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


    # Prediction WITH ADVERSARIAL NOISE
    predictionAdvDeltaAdv = np.random.rand(limit, limit)
    predictionAdvDelta = np.random.rand(limit, limit)
    for t in range(0, len(D[0, :])):
        i = int(D[0, t]) - 1
        j = int(D[1, t]) - 1
        truth = int(D[2, t])

        DeltaI = np.zeros(len(U[:, 0]))
        DeltaJ = np.zeros(len(U[:, 0]))
        DeltaXI = np.zeros(len(sideBilinear[:, 0]))
        DeltaXJ = np.zeros(len(sideBilinear[:, 0]))

        predAdv = (UAdv[:, i].T + DeltaI) @ ULatentScaler @ (UAdv[:, j].T + DeltaJ).T + UBiasAdv[i] + UBiasAdv[j] + (
                    sideBilinear[:, i].T + DeltaXI) @ WBilinearAdv @ (sideBilinear[:, j].T + DeltaXJ).T
        pred = (U[:, i].T + DeltaI) @ ULatentScaler @ (U[:, j].T + DeltaJ).T + UBias[i] + UBias[j] + (
                    sideBilinear[:, i].T + DeltaXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaXJ).T

        sigmaDeltaAdv = 1 / (1 + np.exp(-predAdv))
        sigmaDelta = 1 / (1 + np.exp(-pred))

        # Gamma and DeltaAdv generation for AMF
        GammaIAdv = alpha * (sigmaDeltaAdv - truth) * (ULatentScalerAdv @ (UAdv[:, j].T + DeltaJ).T).T
        GammaJAdv = alpha * (sigmaDeltaAdv - truth) * ((UAdv[:, i].T + DeltaI) @ ULatentScalerAdv)
        GammaXIAdv = alpha * (sigmaDeltaAdv - truth) * (WBilinearAdv @ (sideBilinear[:, j].T + DeltaXJ).T).T
        GammaXJAdv = alpha * (sigmaDeltaAdv - truth) * ((sideBilinear[:, i].T + DeltaXI) @ WBilinearAdv)
        DeltaAdvIAdv = epsilon * GammaIAdv / np.sqrt(max(np.sum(np.power(GammaIAdv, 2)), 0.000001))
        DeltaAdvJAdv = epsilon * GammaJAdv / np.sqrt(max(np.sum(np.power(GammaJAdv, 2)), 0.000001))
        DeltaAdvXIAdv = epsilon * GammaXIAdv / np.sqrt(max(np.sum(np.power(GammaXIAdv, 2)), 0.000001))
        DeltaAdvXJAdv = epsilon * GammaXJAdv / np.sqrt(max(np.sum(np.power(GammaXJAdv, 2)), 0.000001))

        #Gamma and DeltaAdv generation for MF
        GammaI = alpha * (sigmaDelta - truth) * (ULatentScaler @ (U[:, j].T + DeltaJ).T).T
        GammaJ = alpha * (sigmaDelta - truth) * ((U[:, i].T + DeltaI) @ ULatentScaler)
        GammaXI = alpha * (sigmaDelta - truth) * (WBilinear @ (sideBilinear[:, j].T + DeltaXJ).T).T
        GammaXJ = alpha * (sigmaDelta - truth) * ((sideBilinear[:, i].T + DeltaXI) @ WBilinear)
        DeltaAdvI = epsilon * GammaI / np.sqrt(max(np.sum(np.power(GammaI, 2)), 0.000001))
        DeltaAdvJ = epsilon * GammaJ / np.sqrt(max(np.sum(np.power(GammaJ, 2)), 0.000001))
        DeltaAdvXI = epsilon * GammaXI / np.sqrt(max(np.sum(np.power(GammaXI, 2)), 0.000001))
        DeltaAdvXJ = epsilon * GammaXJ / np.sqrt(max(np.sum(np.power(GammaXJ, 2)), 0.000001))

        predictionAdvDeltaAdv[i, j] = (UAdv[:, i].T + DeltaAdvIAdv) @ ULatentScalerAdv @ (UAdv[:, j].T + DeltaAdvJAdv).T + \
                                UBiasAdv[i] + UBiasAdv[j] + (sideBilinear[:, i].T + DeltaAdvXIAdv) @ WBilinearAdv @ (sideBilinear[:, j].T + DeltaAdvXJAdv).T
        predictionAdvDelta[i, j] = (U[:, i].T + DeltaAdvI) @ ULatentScaler @ (U[:, j].T + DeltaAdvJ).T + UBias[i] + UBias[j] + (
                    sideBilinear[:, i].T + DeltaAdvXI) @ WBilinear @ (sideBilinear[:, j].T + DeltaAdvXJ).T

    predictionAdversarialNoiseAdv = 1. / (1 + np.exp(-predictionAdvDeltaAdv))
    predictionAdversarialNoise = 1. / (1 + np.exp(-predictionAdvDelta))

    # Metrics WITHOUT NOISE
    accAdv, recAdv, precAdv, aucAdv, f1Adv = test(predictionAdv, testSet)
    acc, rec, prec, auc, f1 = test(prediction, testSet)

    # Metrics WITH ADVERSARIAL NOISE
    accAdversarialNoiseAdv, recAdversarialNoiseAdv, precAdversarialNoiseAdv, aucAdversarialNoiseAdv, f1AdversarialNoiseAdv = test(predictionAdversarialNoiseAdv, testSet)
    accAdversarialNoise, recAdversarialNoise, precAdversarialNoise, aucAdversarialNoise, f1AdversarialNoise = test(predictionAdversarialNoise, testSet)

    testOnes = []

    for index in range(0, len(testSet[0, :])):
        if testSet[2, index] == 1.0:
            testOnes.append(index)

    # HR WITHOUT NOISE
    hrAdv = hitratio(UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, testSet, testOnes)
    hr = hitratio(U, ULatentScaler, UBias, WBilinear, testSet, testOnes)

    # HR WITH ADVERSARIAL NOISE
    hrAdversarialNoiseAdv, hrAdversarialNoise = hitratioAdversarialNoise(D, U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, sideBilinear, testSet, testOnes, 0)


    print("NO ATTACK")
    print("--- Matrix Factorization ----")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:",f1)
    print("AUC:", auc)
    print("HR@10:", hr)
    print("---- Adversarial Matrix Factorization ----")
    print("Accuracy:", accAdv)
    print("Precision:",precAdv)
    print("Recall:", recAdv)
    print("F1:", f1Adv)
    print("AUC:", aucAdv)
    print("HR@10:", hrAdv)
    print("--------------------------------")
    print("RANDOM ATTACK")
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
    print("--------------------------------")
    print("ADVERSARIAL ATTACK")
    print("--- Matrix Factorization ----")
    print("Accuracy:", accAdversarialNoise)
    print("Precision:", precAdversarialNoise)
    print("Recall:", recAdversarialNoise)
    print("F1:", f1AdversarialNoise)
    print("AUC:", aucAdversarialNoise)
    print("HR@10:", hrAdversarialNoise)
    print("---- Adversarial Matrix Factorization ----")
    print("Accuracy:", accAdversarialNoiseAdv)
    print("Precision:", precAdversarialNoiseAdv)
    print("Recall:", recAdversarialNoiseAdv)
    print("F1:", f1AdversarialNoiseAdv)
    print("AUC:", aucAdversarialNoiseAdv)
    print("HR@10:", hrAdversarialNoiseAdv)

    return


def test(prediction, groundTruth):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    prediction_01 = np.random.rand(len(prediction[:, 0]), len(prediction[:, 0]))

    for i in range(0, len(prediction[0, :])):
        for j in range(0, len(prediction[:, 0])):
            if prediction[i, j] < 0.35:
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




def modelGen(D, U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, sideBilinear, testSet):

    #try:

    master(D, U, ULatentScaler, UBias, WBilinear, UAdv, ULatentScalerAdv, UBiasAdv, WBilinearAdv, sideBilinear, testSet)  # obtain metrics

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


importFile = open("0_file.pkl", "rb")
importSet = open("7_file.pkl", "rb")
model = pickle.load(importFile)
modelTest = pickle.load(importSet)
importFile.close()
importSet.close()

D = np.transpose(np.loadtxt(open("adjancence.csv", "rb"), delimiter=","))

TRAIN_RATIO = 0.9

TestSet = modelTest.testSet

print("Dimension TrainingSet: ", len(D[0, :]))
print("Dimension TestSet: ", len(TestSet[0, :]))

experimentCounter = 0

# 10 latent features for 792 nodes
p = mp.Pool(1)
expcounter = 0

p.apply(modelGen, args=(D, model.U, model.ULatentScaler, model.UBias, model.WBilinear, model.UAdv, model.ULatentScalerAdv, model.UBiasAdv, model.WBilinearAdv, sideBilinear, TestSet))
expcounter += 1
p.close()
p.join()





