import numpy as np
import matplotlib.pyplot as plt
import math

#defining training data
class1 = np.loadtxt("Pierres Mappe/kNN/trainClass1.dat")
class2 = np.loadtxt("Pierres Mappe/kNN/trainClass2.dat")
class3 = np.loadtxt("Pierres Mappe/kNN/trainClass3.dat")
class4 = np.loadtxt("Pierres Mappe/kNN/trainClass4.dat")

#defining test data
unknown = np.loadtxt("Pierres Mappe/kNN/unknown.dat")

#combining data
train_data = [class1, class2, class3, class4]

test_data = unknown

def gaussian_naive_bayes(testdata, traindata):
    
    #initialize list
    priors = []
    means = []
    variances = []

    for cls in train_data:
        #calculate priors
        priors.append(len(cls)/len(train_data))
        
        #calculate means
        means.append(np.mean(cls, axis=0))
        
        #calculate variances
        variances.append(np.var(cls, axis=0))

    priors = np.array(priors)
    priors = priors/np.sum(priors)

    print("Prior probabilities", priors)

    classifications = []

    for observations in test_data:
        scores = []
        for mean, variances, prior in zip(means, variances, priors):
            a = 1/(math.sqrt(np.linalg.det(variances))*(2*math.pi))
            b = np.exp(-0.5*((observation-mean)@(np.linalg.inv(variances))@(observation-mean)))
            result = a*b

            scores.append(math.log(prior)+math.log(result))
        classifications.append(np.argmax(scores))
    
    return classifications

def main():
    class1 = np.loadtxt("Pierres Mappe/kNN/trainClass1.dat")
    class2 = np.loadtxt("Pierres Mappe/kNN/trainClass2.dat")
    class3 = np.loadtxt("Pierres Mappe/kNN/trainClass3.dat")
    class4 = np.loadtxt("Pierres Mappe/kNN/trainClass4.dat")

    train_data = [class1, class2, class3, class4[:5, :]]

    test_data = np.loadtxt("Pierres Mappe/kNN/unknown.dat")

    predictions = gaussian_naive_bayes(test_data, train_data)

main()