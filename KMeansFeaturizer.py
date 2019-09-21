class KMeansFeaturizer:

    def __init__(self, k=10, target_scale=0, random_state=None, n_jobs=-1, n_centroid=5):

        self.k = k
        self.n_jobs = n_jobs
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_centroid=n_centroid

    def fit(self, X, y=None):
        if y is None:
            km_model = KMeans(n_clusters=self.k, n_init=20, random_state=self.random_state,  n_jobs=self.n_jobs)
            km_model.fit(X)
            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self
        data_with_target = np.hstack((X, y[:, np.newaxis] * self.target_scale))
        km_model_pretrain = KMeans(n_clusters=self.k, n_jobs=self.n_jobs, n_init=20, random_state=self.random_state)
        km_model_pretrain.fit(data_with_target)

        km_model = KMeans(n_clusters=self.k, n_jobs=self.n_jobs,
                          init=km_model_pretrain.cluster_centers_[:, :self.n_centroid], n_init=1, max_iter=1)
        km_model.fit(X)
        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self, X, y=None):
        clusters = self.km_model.predict(X)
        return clusters[:, np.newaxis]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
