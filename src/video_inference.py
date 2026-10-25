"""
Batch video anomaly detection inference pipeline.

This module provides complete inference pipeline for video anomaly detection:
1. Processes segmented unseen videos
2. Generates embeddings using trained models (CNN, ResNet, ViT)
3. Passes embeddings through trained SRU/SRU++ models
4. Enhances predictions using cluster-based cosine similarity weighting
5. Outputs final anomaly classification with confidence scores

We implemented this module for real-time or batch inference on video segments
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from video_clustering import VideoEmbeddingClusterer, get_category_names, similarity_weights
from video_embedding_cnn import CNNVideoEmbedder
from video_embedding_resnet import ResNetVideoEmbedder
from video_embedding_vit import ViTVideoEmbedder
from sru_training import SRUModel
from srupp_training import SRUppModel



# Embedder checkpoint written next to the embeddings by video_embedding_*.py (e.g. vit_embedder.pth)
EMBEDDER_PATH = "/path/to/output/embeddings/vit_embedder.pth"
SRU_MODEL_PATH = "models/best_srupp_model.pth"
CLUSTER_MODEL_PATH = "clustering_results/clustering_model.pkl"  # None to disable cluster weighting
VIDEO_PATH = "/path/to/video.mp4"
VIDEO_DIR=None
OUTPUT_DIR = "inference_results"
EMBEDDING_MODEL = "vit"
SRU_MODEL_TYPE = "srupp"
DEVICE = "auto"
CONFIDENCE_THRESHOLD = 0.5
CLUSTER_WEIGHT=0.3
CLUSTER_TEMPERATURE=0.1


EMBEDDER_CLASSES = {
    'cnn': CNNVideoEmbedder,
    'resnet': ResNetVideoEmbedder,
    'vit': ViTVideoEmbedder,
}
SRU_MODEL_CLASSES = {
    'sru': SRUModel,
    'srupp': SRUppModel,
}


class VADInferenceEngine:
    """
    Complete Video Anomaly Detection Inference Engine | Combines video embedding generation, SRU/SRU++ prediction, and
    cluster-based enhancement for robust anomaly detection.
    """

    def __init__( self,
                 device: str = 'auto',
                 embedding_model: str = 'vit',
                 sru_model_type: str = 'srupp',
                 confidence_threshold: float = 0.5,
                 cluster_weight: float = 0.3,
                 cluster_temperature: float = 0.1):
        """
        Initialize the VAD inference engine.

        Args:
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            embedding_model: Type of embedding model ('cnn', 'resnet', 'vit')
            sru_model_type: Type of SRU model ('sru', 'srupp')
            confidence_threshold: Minimum confidence for positive classification
            cluster_weight: Weight for cluster-based enhancement (0.0-1.0)
            cluster_temperature: Softmax temperature applied to the cosine similarities
        """
        self.device = self._setup_device(device)
        self.embedding_model_type = embedding_model.lower()
        self.sru_model_type = sru_model_type.lower()
        self.confidence_threshold=confidence_threshold
        self.cluster_weight = cluster_weight 
        self.cluster_temperature=cluster_temperature

        # Model components
        self.embedding_model=None
        self.sru_model=None
        self.clusterer = None

        # Category mapping (replaced by the trained model's classes in load_sru_model)
        self.category_names = get_category_names(12)

        # Statistics
        self.inference_stats = {
            'total_videos': 0,
            'confidence_scores': [],
        }

        print(f"VAD Inference Engine initialized")
        print(f"Device: {self.device}")
        print(f"Embedding Model: {self.embedding_model_type}")
        print(f"SRU Model: {self.sru_model_type}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'


        torch_device = torch.device(device)
        print(f"Using device: {torch_device}")
        return torch_device



    def load_embedding_model(self, embedder_path: str) -> None:
        """Load the embedder saved with the training embeddings, so inference uses the same weights and sampling."""

        print(f"Loading {self.embedding_model_type.upper()} embedding model...")


        if self.embedding_model_type not in EMBEDDER_CLASSES:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_type}")
        self.embedding_model = EMBEDDER_CLASSES[self.embedding_model_type].from_checkpoint(embedder_path, self.device) 

        print(f"{self.embedding_model_type.upper()} embedding model loaded successfully")

    def load_sru_model(self, model_path: str) -> None:
        """
        Load the trained SRU/SRU++ model.

        Args:
            model_path: Path to trained SRU model
        """
        print(f"Loading {self.sru_model_type.upper()} model from: {model_path}")

        if self.sru_model_type not in SRU_MODEL_CLASSES:
            raise ValueError(f"Unsupported SRU model type: {self.sru_model_type}")

        self.sru_model = SRU_MODEL_CLASSES[self.sru_model_type].from_checkpoint(model_path, self.device)
        self.category_names = get_category_names(self.sru_model.num_classes)
        if self.embedding_model is not None and self.embedding_model.output_dim != self.sru_model.input_size:
            raise ValueError(f"The embedding model produces {self.embedding_model.output_dim}-dim embeddings but the "
                             f"SRU model expects {self.sru_model.input_size}")

        print(f"{self.sru_model_type.upper()} model loaded successfully")
        print(f"Model configuration: {self.sru_model.get_model_info()}")

    def load_cluster_model(self, cluster_path: str) -> None: 

        print(f"Loading cluster model from: {cluster_path}")

        if not cluster_path.endswith('.pkl'):
            # The class centers only make sense together with the scaler/PCA saved in the .pkl model
            raise ValueError(f"Unsupported cluster file format: {cluster_path}. Use clustering_model.pkl")

        self.clusterer = VideoEmbeddingClusterer()
        self.clusterer.load_model(cluster_path) 

        if not self.clusterer.class_centers:
            print("Warning: No class centers found in cluster model")
            self.clusterer = None
        elif self.sru_model is not None and len(self.clusterer.category_names) != self.sru_model.num_classes:
            print(f"Warning: cluster model has {len(self.clusterer.category_names)} classes but the SRU model has "
                  f"{self.sru_model.num_classes}. Skipping cluster enhancement.")
            self.clusterer = None


        print("Cluster model loaded successfully")

    def generate_embedding(self, video_path: str) -> np.ndarray:
        """
        Generate the embedding of a video segment with the same frame sampling used for the training embeddings.

        Args:
            video_path: Path to the video segment

        Returns:
            Video embedding of shape (frames, features)
        """
        if self.embedding_model is None: 
            raise ValueError("Embedding model not loaded. Call load_embedding_model first.")

        return self.embedding_model.process_video(video_path)

    def predict_with_sru(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using SRU/SRU++ model using Video embedding and returns tuple of (logits, probabilities)
        """
        if self.sru_model is None:
            raise ValueError("SRU model not loaded. Call load_sru_model first.")

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float() 


        if len( embedding.shape) == 2:
            embedding = embedding.unsqueeze(1)
        embedding = embedding.to( self.device)

        with torch.no_grad():
            logits = self.sru_model( embedding)
            probabilities=F.softmax(logits, dim=-1)

        return logits.cpu().numpy(), probabilities.cpu().numpy()

    def compute_cluster_enhancement(self,embedding: np.ndarray) -> Optional[np.ndarray]:
        """Class weights from the cosine similarity of the video embedding to each class center, or None."""


        if self.clusterer is None:
            return None 
        similarities = self.clusterer.class_similarities(embedding, len(self.category_names))
        return similarity_weights(similarities,self.cluster_temperature)

    def enhance_predictions(self,
                          sru_probabilities: np.ndarray,
                          cluster_weights: np.ndarray) -> np.ndarray:


        # Combine and enhance SRU predictions using cluster weights.
        enhanced_probs = (1 - self.cluster_weight) * sru_probabilities + \
                        self.cluster_weight * cluster_weights 

        # Renormalize 
        enhanced_probs = enhanced_probs / np.sum(enhanced_probs,axis=-1,keepdims=True)
        return enhanced_probs

    def predict_single_video(self, video_path: str) -> Dict[str, Any]:

        try:
            embedding=self.generate_embedding(video_path)
            sru_logits, sru_probabilities=self.predict_with_sru(embedding)
            cluster_weights = self.compute_cluster_enhancement(embedding)
            if cluster_weights is not None:
                enhanced_probabilities = self.enhance_predictions(sru_probabilities, cluster_weights)
            else:
                enhanced_probabilities = sru_probabilities
            predicted_class = int(np.argmax(enhanced_probabilities[0]))
            confidence=enhanced_probabilities[0, predicted_class] 


            is_anomaly = predicted_class != 0 and confidence >= self.confidence_threshold 
            anomaly_confidence = 1.0 - enhanced_probabilities[0, 0]

            result={
                'video_path': video_path,
                'predicted_class': predicted_class,
                'predicted_category': self.category_names[predicted_class],
                'confidence': float(confidence),
                'is_anomaly': bool(is_anomaly),
                'anomaly_confidence': float(anomaly_confidence),
                'class_probabilities': enhanced_probabilities[0].tolist(),
                'sru_probabilities': sru_probabilities[0].tolist(),
                'cluster_weights': cluster_weights.tolist() if cluster_weights is not None else None,
                'enhancement_applied': cluster_weights is not None
            }
            # Update statistics
            self.inference_stats['total_videos'] += 1
            self.inference_stats['confidence_scores'].append(float(confidence))

            return result



        except Exception as e:
            return {
                'video_path': video_path,
                'error': str(e),
                'success': False
            } 

    def predict_batch(self,video_paths: List[str]) -> List[Dict[str,Any]]:

        results = []

        print(f"Processing {len(video_paths)} videos...")

        for video_path in tqdm(video_paths, desc="Processing videos"):
            result = self.predict_single_video(video_path)
            results.append(result)

        return results

    def get_inference_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:

        successful_results = [r for r in results if 'error' not in r]


        if not successful_results:
            return {
                'error': 'No successful predictions',
                'total_videos': len(results),
                'failed_predictions': len(results),
                'errors': [r['error'] for r in results][:10]
            }

        anomaly_count = sum(1 for r in successful_results if r['is_anomaly'])
        normal_count = len(successful_results) - anomaly_count

        avg_confidence = np.mean([r['confidence'] for r in successful_results])
        avg_anomaly_confidence = np.mean([r['anomaly_confidence'] for r in successful_results])

        class_counts = {}
        for r in successful_results:
            class_name = r['predicted_category']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        summary = {
            'total_videos': len(results),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(results) - len(successful_results),
            'anomaly_count': anomaly_count,
            'normal_count': normal_count,
            'anomaly_rate': anomaly_count / len(successful_results),
            'average_confidence': float(avg_confidence),
            'average_anomaly_confidence': float(avg_anomaly_confidence),
            'class_distribution': class_counts,
            'enhancement_enabled': self.clusterer is not None
        }
        return summary 


def main():
    if not VIDEO_PATH and not VIDEO_DIR:
        raise ValueError("Set either VIDEO_PATH or VIDEO_DIR at the top of this file")

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    # Initialize inference engine
    engine = VADInferenceEngine(
        device = DEVICE, 
        embedding_model=EMBEDDING_MODEL,
        sru_model_type=SRU_MODEL_TYPE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        cluster_weight=CLUSTER_WEIGHT,
        cluster_temperature=CLUSTER_TEMPERATURE
    )

    print("\n" + "="*50)
    print("LOADING MODELS")
    print("="*50)
    engine.load_embedding_model( EMBEDDER_PATH)

    engine.load_sru_model(SRU_MODEL_PATH)

    if CLUSTER_MODEL_PATH:
        engine.load_cluster_model(CLUSTER_MODEL_PATH)

    # Perform inference
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)

    if VIDEO_PATH:
        # Single video inference
        result = engine.predict_single_video(VIDEO_PATH)
        results = [result]
    else:
        # Batch inference
        video_paths = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_paths.extend(Path(VIDEO_DIR).glob(ext))
        video_paths=[str(p) for p in video_paths]

        if not video_paths:
            raise ValueError(f"No video files found in {VIDEO_DIR}")

        results = engine.predict_batch(video_paths)

    results_path = os.path.join(OUTPUT_DIR, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent = 2)



    summary = engine.get_inference_summary(results)
    summary_path = os.path.join(OUTPUT_DIR, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2) 


    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Total videos processed: {summary['total_videos']}")


    if 'error' in summary:
        print(f"All predictions failed. First errors: {summary['errors']}")
        return

    print(f"Anomalies detected: {summary['anomaly_count']}")
    print(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
    print(f"Average confidence: {summary['average_confidence']:.4f}")


if __name__ == "__main__":
    main()
