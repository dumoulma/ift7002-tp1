import file_loader
import numpy as np

import confidence_intervals as ci
from sklearn import neighbors
from sklearn import cross_validation

datasetFuncDict = {"mnist":file_loader.fetch_mnist, "mush":file_loader.fetch_mushroom, "sonar":file_loader.fetch_sonar}

def knn(dataset):
    loadDataset = datasetFuncDict[dataset]
    X_s, y_s, X_t, y_t = loadDataset()
    folds = 10

    score_averages = {}
    print("score_averages", score_averages)
    for n_neighbors in (1, 3, 10):
    #    print("Cas pour n_neighbors=",n_neighbors)
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,)
    
        kf = cross_validation.KFold(X_s.shape[0], n_folds=folds)
        
        scores = []
        for train_index, test_index in kf:
            X_train, X_test = X_s[train_index], X_s[test_index]
            y_train, y_test = y_s[train_index], y_s[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)        
            score = np.sum(y_pred != y_test)
            scores.append(score)
        
        average = float(sum(scores)) / 10
        print("Average:", average)
        score_averages[n_neighbors] = average
    
    bestScore = 100000
    bestN = -1
    for key in score_averages:
        print ("La moyenne de ", key, " est : ", score_averages[key])
        if  score_averages[key] < bestScore:
            bestScore = score_averages[key]
            bestN = key

    print("Le meilleur est", bestN)     
    clf = neighbors.KNeighborsClassifier(n_neighbors=bestN)
    clf.fit(X_s, y_s)
    
    y_pred = clf.predict(X_s)
    Rs = float(np.sum(y_pred != y_s)) / X_s.shape[0]
    print("Le risque empirique Rs est: %f", Rs)
    
    y_pred = clf.predict(X_t)
    Rt = float(np.sum(y_pred != y_t)) / X_t.shape[0]
    print("Le risque empirique Rt est: ", Rt)
    
    n = X_t.shape[0]
    delta = 0.1
    print("Interval de confiance sur Rt:") 
    print("Approximation par la normale): +/- %0.5f"%(ci.confint_normal(n, Rt, delta)))

    print("Approximation par la borne de Hoeffding): +/- %0.5f"%(ci.confint_hoeffding(n, delta)))
    print("Approximation Binomiale): [%0.5f,%0.5f]"%(ci.confint_binomial(n, n*Rt, delta)))
    

# for dataSet in datasetFuncDict:
#    print(dataSet)
#    knn(dataSet)

knn("mnist")
