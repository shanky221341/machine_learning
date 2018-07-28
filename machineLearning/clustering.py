import ggplot
from ggplot import *
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


class Clustering:
    @staticmethod
    def sampleDBSCAN():
        # #############################################################################
        # Generate sample data
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                    random_state=0)
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)

        noise = np.ones_like(db.labels_)
        noise[db.core_sample_indices_] = 0
        a = pd.DataFrame(X
                         )
        a.columns = ['a', 'b']
        a['label'] = labels_true
        a['noise'] = noise

        a['noise'] = a['noise'].astype(str)

        print(X[0:5, ])

        p1 = ggplot(a, aes(x='a', y='b', color="label", shape='noise')) + geom_point()
        return p1

    @staticmethod
    def create_dbscan_cluster(X, labels_true_col):
        data = X.copy()
        X = X.drop(labels_true_col, axis=1)
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        labels = db.labels_
        print(X)
        labels_true = data[labels_true_col]

        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))

        noise = np.ones_like(db.labels_)
        noise[db.core_sample_indices_] = 0
        a = pd.DataFrame(X)
        a.columns = data.columns[0:2]
        a['label'] = labels_true
        a['noise'] = noise
        a['cluster_labels'] = db.labels_

        a['noise'] = a['noise'].astype(str)
        a['label'] = a['label'].astype(str)
        a['cluster_labels'] = a['cluster_labels'].astype(str)

        print(a.dtypes)
        p1 = ggplot(a, aes(x=data.columns[0], y=data.columns[1], color="cluster_labels", shape='noise')) + geom_point()
        return p1
