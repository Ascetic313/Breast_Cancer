# Loading Libraries:
import codecademylib3_seaborn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Loading datasets:
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

# Checking the datasets:
breast_cancer_data.data[0]
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# Splitting the data into Training set:
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
print(len(training_data))
print(len(training_labels))

# Creating a Classifier:
Classifier = KNeighborsClassifier(n_neighbors =3) 
Classifier.fit(training_data, training_labels)

print(Classifier.score(validation_data, validation_labels))

# Alternating a different k:
k_list = []
accuracies = []
for k in range (1, 101):
        classifier = KNeighborsClassifier(n_neighbors= k)
        classifier.fit(training_data, training_labels)
        k_list.append(k)
        accuracies.append(classifier.score(validation_data, validation_labels))
        print(accuracies)

# Graphing different values of k:
plt.plot(k_list, accuracies)
plt.xlabel("Different values of k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
