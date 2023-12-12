#importing libraries
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from KMeans import KMeans
from sklearn.model_selection import train_test_split

def plot_clusters(X, y_means, title):
    colors = ['red', 'blue', 'green', 'yellow','purple']
    unique_labels = set(y_means)
    for label in unique_labels:
        plt.scatter(X[y_means == label, 0], X[y_means == label, 1], color=colors[label])

    plt.title(title)
    plt.show()

def main():
    # Defining the centroids
    num_clusters = 5
    

    # Sample cluster dataset
    X, y = make_blobs(n_samples=200,centers=num_clusters, n_features=2, random_state=2)

    # Splitting the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # Using KMeans
    max_iterations = 500
    km = KMeans(n_clusters=num_clusters, max_iter=max_iterations)
    y_means_train = km.fit_predict(X_train)

   
    plot_clusters(X_train, y_means_train, "Training Set Clusters")

    # Assign clusters to the test set
    y_means_test = km.assign_clusters(X_test)

    # Plotting the points for test set clusters
    plot_clusters(X_test, y_means_test, "Test Set Clusters")

if __name__ == "__main__":
    main()
