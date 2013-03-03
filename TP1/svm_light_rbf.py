import file_loader
import numpy as np
import svmlight
import confidence_intervals as ci
from sklearn import neighbors
from sklearn import cross_validation

datasetFuncDict = {"mnist":file_loader.fetch_mnist_svn, "mush":file_loader.fetch_mushroom_svn, "sonar":file_loader.fetch_sonar_svn}

def svmRBF(dataset):
    loadDataset = datasetFuncDict[dataset]
    S, T = loadDataset()
    folds = 10

    score_averages = {}
    print("score_averages", score_averages)
    for C,gamma in [(C,gamma) for C in (0.1,1,10,100) for gamma in (0.01, 0.1, 1, 10)]:
        kf = cross_validation.KFold(S.shape[0], n_folds=folds)

        scores = []
        for train_index, test_index in kf:
            S_train, S_test = S[train_index], S[test_index]
            
            model = svmlight.learn(S_train, type='classification', C=C, kernel='rbf', rbf_gamma=gamma)

            y_pred = svmlight.classify(model, S_test)
            score = np.sum((y_pred >0.0) != S_test[:,0])             
            scores.append(score)
        
        average = float(sum(scores)) / 10
        print("Average:", average)
        score_averages[(C,gamma)] = average
    
    bestScore = 100000
    bestKey = (-1,-1)
    for key in score_averages:
        print ("La moyenne de ", key, " est : ", score_averages[key])
        if  score_averages[key] < bestScore:
            bestScore = score_averages[key]
            bestKey = key

    print("Le meilleur est ",bestKey[0], " et le meilleur gamma est ",bestKey[1])     
    model = svmlight.learn(S, type='classification', C=bestKey[0], kernel='rbf', rbf_gamma=bestKey[1])
    
    y_pred = svmlight.classify(model, S)
    Rs = float(np.sum((y_pred >0.0) != (S[:,0]))) / S.shape[0]
    print("Le risque empirique Rs est: %f", Rs)
    
    y_pred = svmlight.classify(model, T)
    Rt = float(np.sum((y_pred >0.0) != (T[:,0])))  / T.shape[0]
    print("Le risque empirique Rt est: ", Rt)
    
    n = T.shape[0]
    delta = 0.1
    print("Interval de confiance sur Rt:") 
    print("Approximation par la normale): +/- %0.5f"%(ci.confint_normal(n, Rt, delta)))

    print("Approximation par la borne de Hoeffding): +/- %0.5f"%(ci.confint_hoeffding(n, delta)))
    print("Approximation Binomiale): [%0.5f,%0.5f]"%(ci.confint_binomial(n, n*Rt, delta)))
    

for dataSet in datasetFuncDict:
    print(dataSet)
    svmRBF(dataSet)


