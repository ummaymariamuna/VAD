"""
Video embedding clustering module for video anomaly detection (VAD).
This module provides K-means clustering functionality for video embeddings.
It trains clusters using true labels and embeddings, then creates cluster centers
for enhanced classification in the VAD pipeline.
"""


import os
import numpy as np
import torch
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc 


# Edit these values before running this file.
EMBEDDINGS_PATH = "/path/to/embeddings.npy"
LABELS_PATH = "/path/to/labels.npy"
OUTPUT_DIR = "clustering_results"
N_CLUSTERS = 12 
MAX_ITER = 300
RANDOM_STATE = 42
TEMPORAL_POOLING = "mean"  # "mean" averages the frame embeddings; "flatten" concatenates them
NORMALIZE_EMBEDDINGS=True
USE_PCA = False
PCA_COMPONENTS = 512
VISUALIZE = True
VISUALIZATION_METHOD = "pca"

MULTICLASS_CATEGORY_NAMES = [
    "Normal", "Abuse", "Arson", "Assault", "Road Accident", "Burglary",
    "Explosion", "Fighting", "Robbery", "Shooting", "Stealing", "Vandalism"
]
BINARY_CATEGORY_NAMES = ["Normal", "Anomaly"]


def get_category_names(num_classes: int) -> List[str]:
    """Return display names for the classes of a binary or UCF-Crime multiclass model."""
    if num_classes == len(BINARY_CATEGORY_NAMES):
        return list(BINARY_CATEGORY_NAMES)
    if num_classes == len(MULTICLASS_CATEGORY_NAMES):
        return list(MULTICLASS_CATEGORY_NAMES)
    return [f"Class {i}" for i in range(num_classes)]


def similarity_weights(similarities: np.ndarray,temperature: float=0.1) -> np.ndarray:
    """
    Turn cosine similarities to the class centers into weights that sum to 1.
    Cosine similarities lie in [-1, 1], so a plain softmax would give almost uniform weights; the temperature sharpens
    them. Classes without a center (NaN similarity) get weight 0.
    """
    similarities=np.asarray(similarities,dtype=np.float64)
    available = ~np.isnan(similarities)
    weights = np.zeros_like(similarities)
    if not available.any():
        return weights
    scaled = similarities[available] / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    weights[available] = exp_scaled / np.sum(exp_scaled)
    return weights


class VideoEmbeddingClusterer:
    """
    K-means clustering for video embeddings with multiple similarity metrics.
    """

    def __init__(self,
                 n_clusters: int = 12,
                 random_state: int = 42,
                 max_iter: int = 300, 
                 temporal_pooling: str = 'mean',
                 normalize_embeddings: bool=True,
                 use_pca: bool = False, 
                 pca_components: int=512):

        if temporal_pooling not in ('mean', 'flatten'):
            raise ValueError(f"Unsupported temporal pooling: {temporal_pooling}")


        self.n_clusters = n_clusters 
        self.random_state = random_state
        self.max_iter = max_iter
        self.temporal_pooling = temporal_pooling
        self.normalize_embeddings = normalize_embeddings
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.kmeans = None
        self.scaler = None 
        self.pca = None
        self.cluster_centers = None
        self.class_centers = None 
        self.cluster_labels = None
        self.embedding_shape=None 

        self.category_names = list(MULTICLASS_CATEGORY_NAMES)

        self.training_stats = {}

        print(f"VideoEmbeddingClusterer initialized with {n_clusters} clusters")

    def _set_category_names(self, labels: np.ndarray) -> None:
        num_classes = max(int(np.max(labels)) + 1, 2)
        self.category_names=get_category_names(num_classes)

    def prepare_embeddings(self, embeddings: np.ndarray, verbose: bool=True) -> np.ndarray:
        """
        Map video embeddings of shape (videos, frames, features) into the space the clusters and class centers live in.
        Inference must use this same method so its embeddings are pooled, scaled and projected identically.
        """

        if verbose:
            print(f"Original embeddings shape: {embeddings.shape}")
        self.embedding_shape = self.embedding_shape or embeddings.shape

        if self.temporal_pooling == 'mean' and embeddings.ndim == 3:
            pooled_embeddings=embeddings.mean(axis=1)
        else:
            pooled_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        if verbose:
            print(f"Pooled embeddings shape: {pooled_embeddings.shape}")
        if self.normalize_embeddings:
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized_embeddings=self.scaler.fit_transform(pooled_embeddings)
                print("Applied StandardScaler normalization")
            else:
                normalized_embeddings = self.scaler.transform(pooled_embeddings)
        else:
            normalized_embeddings = pooled_embeddings

        if self.use_pca:
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                pca_embeddings=self.pca.fit_transform(normalized_embeddings)
                print(f"Applied PCA: {normalized_embeddings.shape} -> {pca_embeddings.shape}")
                print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
            else:
                pca_embeddings=self.pca.transform(normalized_embeddings)

            return pca_embeddings

        return normalized_embeddings


    def train_clusters(self,
                      embeddings: np.ndarray,
                      labels: np.ndarray) -> Dict[str, Any]:

        labels=np.asarray( labels).astype(np.int64)
        self._set_category_names(labels)

        print("Starting cluster training...")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")

        processed_embeddings = self.prepare_embeddings(embeddings)

        self.kmeans=KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter = self.max_iter,
            n_init=10
        )

        print("Training K-means clustering...")
        cluster_predictions = self.kmeans.fit_predict(processed_embeddings)

        self.cluster_centers=self.kmeans.cluster_centers_
        self.cluster_labels = cluster_predictions

        silhouette_avg = silhouette_score(processed_embeddings, cluster_predictions)
        ari_score = adjusted_rand_score(labels, cluster_predictions)
        inertia = self.kmeans.inertia_

        cluster_label_analysis = self._analyze_cluster_label_correspondence(
            cluster_predictions, labels
        )


        label_to_cluster_map = self._create_label_cluster_mapping(
            cluster_predictions, labels
        )

        self.training_stats={
            'silhouette_score': silhouette_avg,
            'adjusted_rand_index': ari_score,
            'inertia': inertia,
            'n_clusters': self.n_clusters,
            'n_samples': len(embeddings),
            'cluster_label_analysis': cluster_label_analysis,
            'label_to_cluster_map': label_to_cluster_map
        }

        print(f"Clustering completed!")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Adjusted Rand Index: {ari_score:.4f}")
        print(f"Inertia: {inertia:.2f}")

        return self.training_stats

    def _analyze_cluster_label_correspondence(self, 
                                           cluster_predictions: np.ndarray, 
                                           true_labels: np.ndarray) -> Dict[str,Any]:

        analysis={}

        confusion_matrix = np.zeros((self.n_clusters, len(self.category_names)))



        for cluster_id in range(self.n_clusters):
            cluster_mask=cluster_predictions == cluster_id
            cluster_true_labels = true_labels[cluster_mask]

            for label in cluster_true_labels:
                if 0 <= label < len(self.category_names):
                    confusion_matrix[cluster_id, int( label)] += 1
        analysis['confusion_matrix'] = confusion_matrix


        cluster_best_labels = {}
        for cluster_id in range( self.n_clusters):
            cluster_size=np.sum(confusion_matrix[cluster_id, :])
            if cluster_size == 0:
                continue
            best_label = np.argmax(confusion_matrix[cluster_id, :])
            confidence = confusion_matrix[cluster_id,best_label] / cluster_size
            cluster_best_labels[cluster_id]={
                'best_label': int(best_label),
                'confidence': float(confidence),
                'category_name': self.category_names[best_label]
            }

        analysis['cluster_best_labels'] = cluster_best_labels


        return analysis

    def _create_label_cluster_mapping(self,
                                    cluster_predictions: np.ndarray,
                                    true_labels: np.ndarray) -> Dict[int,int]:
        label_cluster_map = {}

        for label in range( len(self.category_names)):
            label_mask=true_labels == label
            if np.sum(label_mask) > 0:
                label_clusters=cluster_predictions[label_mask]
                most_common_cluster = np.bincount(label_clusters).argmax()
                label_cluster_map[label]=int(most_common_cluster)

        return label_cluster_map

    def compute_class_centers(self,
                            embeddings: np.ndarray,
                            labels: np.ndarray) -> Dict[int, np.ndarray]:

        labels = np.asarray(labels).astype(np.int64) 
        self._set_category_names(labels)
        print("Computing class centers...")
        processed_embeddings = self.prepare_embeddings(embeddings)

        class_centers = {}
        for class_id in range(len(self.category_names)):
            class_mask = labels == class_id
            if np.sum(class_mask) > 0:
                class_embeddings=processed_embeddings[class_mask] 
                class_center = np.mean(class_embeddings, axis=0)
                class_centers[class_id] = class_center
                print(f"Class {class_id} ({self.category_names[class_id]}): {np.sum(class_mask)} samples")
            else:
                print(f"Warning: No samples found for class {class_id} ({self.category_names[class_id]})")

        self.class_centers = class_centers
        return class_centers

    def class_similarities(self, embedding: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """
        Cosine similarity between one video embedding of shape (frames, features) and every class center.
        Returns an array of length num_classes; classes without a center are NaN.
        """
        if not self.class_centers:
            raise ValueError("Class centers not computed. Call compute_class_centers first.")


        num_classes = num_classes or len(self.category_names) 
        processed_embedding = self.prepare_embeddings(np.asarray(embedding)[np.newaxis], verbose=False)
        similarities = np.full(num_classes,np.nan)
        for class_id, center in self.class_centers.items():
            if class_id < num_classes:
                similarities[class_id]=cosine_similarity(processed_embedding, center.reshape(1, -1))[0][0]

        return similarities

    def predict_clusters( self,embeddings: np.ndarray) -> np.ndarray:

        if self.kmeans is None:
            raise ValueError("Model not trained. Call train_clusters first.")

        processed_embeddings=self.prepare_embeddings(embeddings) 
        return self.kmeans.predict(processed_embeddings)

    def compute_cosine_similarities( self,
                                  embedding: np.ndarray,
                                  centers: Dict[int, np.ndarray]) -> Dict[int, float]:

        similarities = {} 

        for class_id, center in centers.items():
            similarity = cosine_similarity([embedding], [center])[0][0] 
            similarities[class_id] = float(similarity)

        return similarities

    def visualize_clusters(self,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          save_path: Optional[str] = None,
                          method: str = 'pca') -> None:

        print(f"Visualizing clusters using {method.upper()}...")

        labels=np.asarray(labels).astype(np.int64)
        processed_embeddings = self.prepare_embeddings(embeddings) 

        if method.lower() == 'pca':
            reducer = PCA( n_components=2, random_state=self.random_state)
            embeddings_2d = reducer.fit_transform(processed_embeddings)
            centers_2d=reducer.transform( self.cluster_centers) if self.cluster_centers is not None else None
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
            if self.cluster_centers is not None:
                combined_2d=reducer.fit_transform(np.vstack([processed_embeddings,self.cluster_centers]))
                embeddings_2d = combined_2d[:len(processed_embeddings)]
                centers_2d = combined_2d[len(processed_embeddings):] 
            else:
                embeddings_2d = reducer.fit_transform(processed_embeddings) 
                centers_2d = None

        plt.figure( figsize = (15, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.category_names)))

        for class_id in range(len(self.category_names)):
            class_mask = labels == class_id
            if np.sum(class_mask) > 0:
                plt.scatter(
                    embeddings_2d[class_mask, 0],
                    embeddings_2d[class_mask, 1],
                    label=self.category_names[class_id],
                    alpha=0.6,
                    s=50,
                    color=colors[class_id]
                ) 

        if centers_2d is not None:
            plt.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                marker='x',
                color='red',
                s=200,
                linewidth = 3,
                label='Cluster Centers'
            )

        plt.title(f'Video Embedding Clusters ({method.upper()} Visualization)')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid( True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")


        plt.show( ) 

    def save_model( self,save_path: str) -> None:


        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_centers': self.cluster_centers,
            'class_centers': self.class_centers,
            'cluster_labels': self.cluster_labels,
            'embedding_shape': self.embedding_shape,
            'training_stats': self.training_stats,
            'config': {
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'max_iter': self.max_iter,
                'temporal_pooling': self.temporal_pooling,
                'normalize_embeddings': self.normalize_embeddings,
                'use_pca': self.use_pca,
                'pca_components': self.pca_components
            }, 
            'category_names': self.category_names
        }
        with open(save_path, 'wb') as f:
            pickle.dump(model_data,f)

        print(f"Model saved to: {save_path}")

    def load_model( self,load_path: str) -> None:

        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)

        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.cluster_centers = model_data['cluster_centers']
        self.class_centers = model_data.get('class_centers')
        self.cluster_labels = model_data['cluster_labels']
        self.embedding_shape = model_data['embedding_shape']
        self.training_stats = model_data['training_stats']
        self.category_names = model_data['category_names']

        config = model_data['config']
        self.n_clusters = config['n_clusters']
        self.random_state = config['random_state']
        self.max_iter = config['max_iter']
        #Models saved before temporal pooling existed flattened the frame embeddings
        self.temporal_pooling = config.get('temporal_pooling', 'flatten')
        self.normalize_embeddings = config['normalize_embeddings']
        self.use_pca = config['use_pca']
        self.pca_components = config['pca_components']

        print(f"Model loaded from: {load_path}")
        print(f"Loaded model with {self.n_clusters} clusters")


def load_embeddings_and_labels(embeddings_path: str,  
                              labels_path: str) -> Tuple[np.ndarray,np.ndarray]:


    print(f"Loading embeddings from: {embeddings_path}")
    print(f"Loading labels from: {labels_path}")


    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path).astype(np.int64)

    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    return embeddings, labels




def main():
    #Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    embeddings, labels = load_embeddings_and_labels(EMBEDDINGS_PATH, LABELS_PATH)

    clusterer = VideoEmbeddingClusterer(
        n_clusters=N_CLUSTERS, 
        random_state = RANDOM_STATE,
        max_iter=MAX_ITER,
        temporal_pooling=TEMPORAL_POOLING,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        use_pca=USE_PCA,
        pca_components=PCA_COMPONENTS
    )

    print("\n" + "="*50)
    print("TRAINING CLUSTERS")
    print("="*50)
    training_stats = clusterer.train_clusters(embeddings, labels)

    print("\n" + "="*50)
    print("COMPUTING CLASS CENTERS")
    print("="*50)
    class_centers=clusterer.compute_class_centers(embeddings, labels)

    model_path = os.path.join(OUTPUT_DIR, 'clustering_model.pkl')
    clusterer.save_model(model_path) 
    centers_path = os.path.join(OUTPUT_DIR, 'class_centers.npy')
    centers_dict_path = os.path.join(OUTPUT_DIR, 'class_centers.pkl')

    center_dim = next(iter(class_centers.values())).shape[0]
    centers_array = np.array([class_centers.get(i, np.zeros(center_dim))
                             for i in range(len(clusterer.category_names))])
    np.save(centers_path, centers_array)

    with open(centers_dict_path, 'wb') as f:
        pickle.dump(class_centers, f)

    print(f"Class centers saved to: {centers_path}")
    print(f"Class centers dict saved to: {centers_dict_path}")


    # Save training statistics
    stats_path = os.path.join(OUTPUT_DIR, 'clustering_stats.json')
    with open(stats_path, 'w') as f:
        json_stats = {}
        for key,value in training_stats.items():
            if isinstance(value, np.ndarray):
                json_stats[key] = value.tolist()
            elif isinstance(value, dict):
                json_stats[key] = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k,v in value.items()}
            else:
                json_stats[key]=value
        json.dump(json_stats, f, indent=2)

    print(f"Training statistics saved to: {stats_path}")

    # Create visualizations if requested
    if VISUALIZE:
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)

        vis_path = os.path.join(OUTPUT_DIR, f'cluster_visualization_{VISUALIZATION_METHOD}.png')
        clusterer.visualize_clusters(embeddings, labels, vis_path, VISUALIZATION_METHOD)

    print("\n" + "="*50)
    print("CLUSTERING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Silhouette Score: {training_stats['silhouette_score']:.4f}")
    print(f"Adjusted Rand Index: {training_stats['adjusted_rand_index']:.4f}")


if __name__ == "__main__":
    main() 
