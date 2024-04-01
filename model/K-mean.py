import sys
sys.path.append('../')

from data.get_data import get_data
from sklearn.cluster import KMeans
import numpy as np


X, y = get_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
centroids = kmeans.cluster_centers_

# Gán nhãn của các cụm cho từng điểm dữ liệu
labels = kmeans.labels_


# labels chứa nhãn của từng điểm dữ liệu
# Đếm số lượng điểm trong từng cụm bằng cách đếm số lượng điểm có nhãn tương ứng
unique, counts = np.unique(labels, return_counts=True)

# In ra số lượng của từng cụm
for cluster, count in zip(unique, counts):
    print(f"Cụm {cluster}: {count} điểm")
