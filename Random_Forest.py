import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

INPUT_PATH = "../inputs/breast-cancer-wisconsin.data"
OUTPUT_PATH = "../inputs/breast-cancer-wisconsin.csv"
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]
def read_data(path):
    data = pd.read_csv(path)
    return data

def get_headers(dataset):

    return dataset.columns.values

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset


def data_file_to_csv():

    headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
               "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
               "CancerType"]
    # Load the dataset into Pandas data frame]
    dataset = read_csv(INPUT_PATH)

    dataset = add_headers(dataset, headers)

    dataset.to_csv(OUTPUT_PATH, index=False)
    print "File saved...!"


def handel_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

def split_dataset(dataset, train_percentage, feature_headers, target_header):

    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):

    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def dataset_statistics(dataset):
    print dataset.describe()

def main():

    dataset = pd.read_csv(OUTPUT_PATH)
    dataset_statistics(dataset)
    dataset = handel_missing_values(dataset, HEADERS[6], '?')
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1] )

    trained_model = random_forest_classifier(train_x, train_y)
    print "Trained model ::", trained_model

    predictions = trained_model.predict(test_x)

    for i in xrange(0, 5):
        print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])
    print "Train Accuracy:: ", accuracy_score(train_y, trained_model.predict(train_x))
    print "Test Accuracy:: ", accuracy_score(test_y, predictions)


if __name__ == "__main__":
    main()
