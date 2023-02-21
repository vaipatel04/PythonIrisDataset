import numpy as np
import pandas as pd
from pandas import DataFrame
from statistics import mode
import matplotlib.pyplot as plt

#ML classification problem: solve it using k-nearest neighbourhood algorithm
#knn: allows you to control how many nearby objects you look at and compare the object too

def fetchData() -> DataFrame:
    data = pd.read_csv('IRIS.csv')  # reads the file from pandas
    return data

# This function is responsible for renaming columns in the IRIS data set
# as it makes it easier to work with.
# It takes in the iris data set in a DATA FRAME
# it returns the CLEANED DATA FRAME
def cleanData(data: DataFrame): #data: dataframe --> implies that the data has to be a data frame
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data.rename(columns = {cols[0]:0, cols[1]:1, cols[2]:2, cols[3]:3}, inplace=True)
    # print (data.loc[::50]) this prints every 50th line
    data['distance'] = 9999 #distance away from the objects
    return data

def target():#create a target item so we can compare to each one of these items the dataset and figure out the distance; pick sme random values, put it into a list and pass it into a series constructor so you compare the true distance
    targetS = pd.Series([7.0, 3.1, 5.6, 1.9])
    return targetS

#use euclidean distance formula to calc the distance between 2 points. This formula is used when there are multiple attributes to compare objects.
#Euclidean distance formula for between 2 points: d(p, q): sqrt (sum((q-p)^2))); in this case its between the target output from the prev. function i.e., i=0 to i=1
def distance(data):
    targetS = target()
    data['distance'] = (((data.loc[:, 0]-targetS[0])**2)+
                        ((data.loc[:, 1]-targetS[1])**2)+
                        ((data.loc[:, 2]-targetS[2])**2)+
                        ((data.loc[:, 3]-targetS[3])**2))**(1/2)
    return data

#prediction: iris-virginica has the closest neighbours, now set k = 7 and look at the 7 nearest neighbours
def sort_data(data: DataFrame):
    return data.sort_values('distance', ascending=True)
def knn(sorted):
    k = 7
    knn = list(sorted.head(k).species)
    return knn

#use mode to show what the nearest variety of neighbours
def knn_mode(knn):
    return mode(knn)

def knn_scatter_plot(sorted_scatter):
    targetS = target()
    colours = {'Iris-setosa':'red', 'Iris-virginica': 'blue', 'Iris-versicolor': 'green'}
    plt.scatter(
        sorted_scatter[2],
        sorted_scatter[3],
        c=sorted_scatter['species'].map(colours))
    plt.scatter(targetS[2], targetS[3], c='orange')
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.title("Iris Data Scatter Plot")
    plt.show()

if __name__ == '__main__':
    data = fetchData()
    cleanedData = cleanData(data)
    #print(cleanedData)
    target()
    calc_distance = distance(cleanedData)
    #print(calc_distance)
    sorted_data = sort_data(calc_distance)
    #print(sorted_data)
    knn_mode_data = knn(sorted_data)
    #print(knn_mode(knn_mode_data))
    print(knn_scatter_plot(sorted_data))