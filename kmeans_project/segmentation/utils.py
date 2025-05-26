import numpy as np
import cv2
import time
from skimage import color
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin

class KMeans:
    def __init__(self, k, max_iters=50, tol=1e-2, init_method="kmeans++"):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _assign_clusters(self, X):
        start_time = time.time()
        labels = pairwise_distances_argmin(X, self.centroids)
        print(f"_assign_clusters завершён за {time.time() - start_time:.2f} секунд")
        return labels

    def _initialize_centroids(self, X):
        start_time = time.time()
        if self.init_method == "random":
            indices = np.random.choice(X.shape[0], self.k, replace=False)
            centroids = X[indices]
        elif self.init_method == "kmeans++":
            n_samples = X.shape[0]
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(1, self.k):
                distances = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
                probs = distances / distances.sum()
                cumulative_probs = np.cumsum(probs)
                r = np.random.random()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[j])
                        break
            centroids = np.array(centroids)
        print(f"_initialize_centroids завершён за {time.time() - start_time:.2f} секунд")
        return centroids

    def fit(self, X):
        start_time = time.time()
        self.centroids = self._initialize_centroids(X)
        for i in range(self.max_iters):
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)
            for j in range(self.k):
                if np.sum(self.labels == j) > 0:
                    self.centroids[j] = np.mean(X[self.labels == j], axis=0)
            if np.sum((self.centroids - old_centroids) ** 2) < self.tol:
                print(f"KMeans завершён на итерации {i+1} за {time.time() - start_time:.2f} секунд")
                break
        self.inertia_ = np.sum((X - self.centroids[self.labels]) ** 2)
        print(f"fit завершён за {time.time() - start_time:.2f} секунд")
        return self

def find_optimal_k(image_data, max_k=10, min_k=3):
    start_time = time.time()
    silhouette_scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(k=k, init_method="kmeans++")
        kmeans.fit(image_data)
        if len(np.unique(kmeans.labels)) > 1:
            score = silhouette_score(image_data, kmeans.labels, sample_size=1000)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)
        print(f"find_optimal_k: k={k}, время={time.time() - start_time:.2f} секунд")
    optimal_k = np.argmax(silhouette_scores) + min_k
    print(f"Оптимальное k: {optimal_k}, силуэтные коэффициенты: {silhouette_scores}, всего {time.time() - start_time:.2f} секунд")
    return optimal_k

def segment_image(image_path, k=None, output_path=None):
    start_time = time.time()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Сжимаем изображение до максимального размера 512x512
    max_size = 512
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        print(f"Изображение сжато до {image.shape[:2]}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    image_data = image_rgb.reshape(-1, 3).astype(np.float32)
    print(f"image_data shape: {image_data.shape}, время подготовки={time.time() - start_time:.2f} секунд")
    
    if k is None:
        k = find_optimal_k(image_data)
        print(f"Выбрано k={k} в автоматическом режиме")
    
    kmeans = KMeans(k=k, init_method="kmeans++")
    kmeans.fit(image_data)
    
    labels = kmeans.labels.reshape(height, width)
    centers = kmeans.centroids.astype(np.uint8)
    
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(height, width, 3)
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, segmented_image_bgr)
    
    print(f"segment_image завершён за {time.time() - start_time:.2f} секунд")
    return segmented_image, labels, centers, (height, width), segmented_image_bgr, image_rgb