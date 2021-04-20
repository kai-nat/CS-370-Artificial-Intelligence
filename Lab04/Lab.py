from sklearn.neighbors import KNeighborsClassifier
import time
import numpy as np
import pandas as pd
from scipy.stats import mode

dataset = pd.read_csv('TrainingSet.csv')
y_train = dataset['plant'].values
x_train = dataset[['leaf.length', 'leaf.width', 'flower.length', 'flower.width']].values

def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

def predict(x_train, y , x_input, k):
    op_labels = []
    for item in x_input: 
        point_dist = []
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            #Calculating the distance
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
        dist = np.argsort(point_dist)[:k] 
        labels = y[dist]
        lab = mode(labels).mode[0]
        op_labels.append(lab)
    return op_labels

test_data = pd.read_csv('TestSet1.csv')
x_test = test_data[['leaf.length', 'leaf.width', 'flower.length', 'flower.width']].values

k_values = [3,5,7]
prediction = []
times = {}

for i in k_values:
    #predictions from self made formula
    start_time = time.time()
    y_test = predict(x_train,y_train,x_test , i)
    times['Time for k = ' + str(i) + ' (Self)'] = (time.time() - start_time)
    
    prediction.append(y_test)
        
    #predictions from scikit learn
    classifier = KNeighborsClassifier(n_neighbors=i)
    start_time = time.time()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    times['Time for k = ' + str(i) + ' (Scikit)'] = (time.time() - start_time)
    
    prediction.append(list(y_pred))

result = pd.concat([pd.DataFrame(x_test), pd.DataFrame(prediction[0]), pd.DataFrame(prediction[1]), pd.DataFrame(prediction[2]), pd.DataFrame(prediction[3]), pd.DataFrame(prediction[4]), pd.DataFrame(prediction[5])], axis=1)
result.columns = ['leaf.length', 'leaf.width', 'flower.length', 'flower.width','k3','k3_scikit','k5','k5_scikit','k7','k7_scikit']
print(result)

print("\nTime for prediction by both methods:")
print("______________________________________")
for i,j in times.items():
    print(i + " " + str(j))

print("\nCheck Predictions")
print("___________________")
precentage = (sum(((result['k3'] == result['k3_scikit']) == (result['k5'] == result['k5_scikit'])) == (result['k7'] == result['k7_scikit']))/30)*100
print(str(precentage) + '% of entries are equal')

result.to_csv('result.csv', index=False)