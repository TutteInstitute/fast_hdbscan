import fast_hdbscan
from sklearn.datasets import make_blobs

data, _ = make_blobs(1000)

clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data)

