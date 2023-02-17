
import numpy as np
import random as rd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt  # sees how the data is spreadout
from pandas import DataFrame

# analyze the dat: how the data is dispersed across the attributes
# and then classify the data; k-clustering

# this function reads data from iris csv file
# this function takes nothing and return the iris dataset in a dataframe
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
    return data

# this function is responsible for printing the data.
# It takes a data frame as input
# it returns NOTHING
def descriptiveAnalysis(data: DataFrame):
    print(data.shape) # prints the mxn of the dataframe
    print(data.describe()) # descriptive analysis of the data

# this function counts how many each we have of species
def valueCounts(data):
    return data.species.value_counts() # this is the same as data['species'].value_counts()

# This function plots graphs of the columns from the dataframe, specifcially onto one chart
#def plots(data: DataFrame):
#    bins = np.linspace(0, 8, 100)
#    plt.hist(data[0], bins, label="sepal length")
#    plt.hist(data[1], bins, alpha=1, label="sepal width")
#    plt.hist(data[2], bins, alpha=0.75, label="petal length")
#    plt.hist(data[3], bins, alpha=0.5, label="petal width")
#    plt.legend(loc='upper right')
#    plt.show()

#This function plots histograms onto different charts as a subplot
def histplots(data:DataFrame):
    fig, ax = plt.subplots(2, 2, figsize=(8, 4))
    ax[0, 0].hist(data[0])
    ax[0, 1].hist(data[1])
    ax[1, 0].hist(data[2])
    ax[1, 1].hist(data[3])
    ax[0, 0].set_title("sepal length")
    ax[0, 1].set_title("sepal width")
    ax[1, 0].set_title("petal length")
    ax[1, 1].set_title("petal width")
    plt.show()

def scatterplots(data:DataFrame):
    colours = {'Iris-setosa': 'red', 'Iris-virginica': 'blue', 'Iris-versicolor': 'green'} #create a dictionary for colours so we can visualize the scatter plot better
    plt.scatter(
        data[2],
        data[3],
        c=data['species'].map(colours)
    )
    plt.scatter(
        data[0],
        data[1],
        c=data['species'].map(colours)
    )
    plt.show()

def correlation(data:DataFrame):
    print(data.corr(data))

if __name__ == '__main__':
    data = fetchData()
    cleanedData = cleanData(data)
    descriptiveAnalysis(cleanedData)
    #print(x) # shows 4 attributes, and species; we dont need the ID tho
    #print(valueCounts(cleanedData))
    #print(histplots(cleanedData))
    #print(scatterplots(cleanedData))
    correlation(cleanedData)

