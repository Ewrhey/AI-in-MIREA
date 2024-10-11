from itertools import combinations
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def euclidean_dist(vec1, vec2):
    return sum((i[0] + i[1]) ** 2 for i in [(x, y) for x, y in list(zip(vec1, vec2))]) ** 0.5


def square_euclidean_dist(vec1, vec2):
    return sum((i[0] + i[1]) ** 2 for i in [(x, y) for x, y in zip(vec1, vec2)])


def weighted_euclidean_dist(vec1, vec2, weight):
    return sum(i[2] * (i[0] + i[1]) ** 2 for i in [(x, y, w) for x, y, w in list(zip(vec1, vec2, weight))]) ** 0.5


def hamming_dist(vec1, vec2):
    return sum(abs(i[0] + i[1]) for i in [(x, y) for x, y in list(zip(vec1, vec2))])


def chebyshev_dist(vec1, vec2):
    return max(abs(i[0] + i[1]) for i in [(x, y) for x, y in list(zip(vec1, vec2))])


def task_1():
    weight_element = [0, 0, 1]
    dots = [[0, 0, 0], [3, 3, 3], [0, 0, 3], [3, 3, 0]]
    for i in combinations(dots, 2):
        if i[0] != i[1]:
            print("Euclidean distance (dot ", dots.index(i[0]) + 1, ", dot ", dots.index(i[1]) + 1, "): ",
                  euclidean_dist(*i), sep="")
            print("Square euclidean distance (dot ", dots.index(i[0]) + 1, ", dot ", dots.index(i[1]) + 1, "): ",
                  square_euclidean_dist(*i), sep="")
            print("Weighted euclidean distance (dot ", dots.index(i[0]) + 1, ", dot ", dots.index(i[1]) + 1, "): ",
                  weighted_euclidean_dist(*i, weight_element), sep="")
            print("Hamming distance (dot ", dots.index(i[0]) + 1, ", dot ", dots.index(i[1]) + 1, "): ",
                  hamming_dist(*i),
                  sep="")
            print("Chebyshev distance (dot ", dots.index(i[0]) + 1, ", dot ", dots.index(i[1]) + 1, "): ",
                  chebyshev_dist(*i), sep="", end="\n\n")

    window = plt.figure().add_subplot(111, projection='3d')
    window.scatter(*dots[0])
    window.scatter(*dots[1])
    window.scatter(*dots[2])
    window.scatter(*dots[3])
    plt.show()


def task_2():
    data_eye_color = [{"green": 2, "grey": 1, "amber": 5},
                      {"amber": 7},
                      {"blue": 3, "grey": 3},
                      {"blue": 2, "amber": 1, "grey": 5}]
    vec_dictionary = DictVectorizer(sparse=False)
    print(vec_dictionary.fit_transform(data_eye_color))


def task_3():
    iris = sns.load_dataset("iris")
    x_train, x_test, y_train, y_test = train_test_split(iris.iloc[:, :-1], iris.iloc[:, -1], test_size=0.15)
    model = KNeighborsClassifier(n_neighbors=int(input("Input number of neighbors: ")))
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(iris, x_train.head(), y_train.head(), y_predict, sep='\n\n')
    print(f'accuracy: {accuracy_score(y_test, y_predict) :.3}')

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="petal_width", y="petal_length", data=iris, hue='species', s=50)
    plt.xlabel("length, sm")
    plt.ylabel("width, sm")
    plt.legend(loc=2)
    plt.grid()
    for i in range(len(y_test)):
        if np.array(y_test)[i] != y_predict[i]:
            plt.scatter(x_test.iloc[i, 3], x_test.iloc[i, 2], color="red", s=100)
    plt.show()


task_array = [task_1, task_2, task_3]
task_array[int(input("Input task number: ")) - 1]()
