"""
Real-time video anomaly detection stream processor.
This module provides a complete real-time video stream processing pipeline for anomaly detection:
1. Continuous video stream buffering with queue-based frame management
2. Segment-based processing (segments sampled exactly like the training clips)
3. Video embedding generation using CNN, ResNet50, or ViT models
4. SRU/SRU++ temporal sequence modeling
5. Cluster-based enhancement with cosine similarity weighting
6. Real-time anomaly classification with confidence scores
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import pickle
import json
import threading
import time
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from pathlib import Path
import logging
from queue import Queue,Empty
import warnings
warnings.filterwarnings('ignore')

from video_clustering import VideoEmbeddingClusterer,get_category_names, similarity_weights
from video_embedding_cnn import CNNVideoEmbedder
from video_embedding_resnet import ResNetVideoEmbedder
from video_embedding_vit import ViTVideoEmbedder
from sru_training import SRUModel
from srupp_training import SRUppModel


EMBEDDER_CLASSES = {
    'cnn': CNNVideoEmbedder,
    'resnet': ResNetVideoEmbedder,
    'vit': ViTVideoEmbedder,
}

class FrameBuffer:
    """
    Thread-safe circular buffer for continuous video stream management.
    Acts as a queue that holds N seconds of video footage at specified FPS.
    """

    def __init__(self,
                 buffer_duration: int = 30,
                 fps: int = 30,
                 max_frame_size: Tuple[int, int] = (224, 224)):  # (width, height)


        self.buffer_duration = buffer_duration
        self.fps = fps
        self.max_frames=buffer_duration * fps
        self.max_frame_size = tuple( max_frame_size)

        self.buffer = deque(maxlen = self.max_frames)
        self.lock = threading.Lock()

        self.total_frames_added = 0
        self.frames_dropped = 0

        print(f"FrameBuffer initialized: {buffer_duration}s @ {fps}fps = {self.max_frames} frames")
    def add_frame(self, frame: np.ndarray) -> bool:
        try:
            # Resize frame if necessary (frame.shape is (height, width), the target size is (width, height))
            if (frame.shape[1], frame.shape[0]) != self.max_frame_size:
                frame=cv2.resize(frame, self.max_frame_size)

            with self.lock:
                if len(self.buffer) >= self.max_frames:
                    self.frames_dropped += 1

                self.buffer.append(frame.copy())
                self.total_frames_added += 1

            return True

        except Exception as e:
            print(f"Error adding frame to buffer: {e}")
            return False

    def get_segment(self, segment_size: int = 100) -> Optional[np.ndarray]:

        with self.lock:
            if len(self.buffer) < segment_size:
                return None

            segment_frames = list(self.buffer)[-segment_size:]
            segment = np.array(segment_frames)

            return segment

    def get_latest_frames(self,count: int) -> Optional[np.ndarray]:

        with self.lock:
            if len( self.buffer) < count:
                return None

            latest_frames = list(self.buffer)[-count:]
            return np.array( latest_frames)

    def pop_segment( self, segment_size: int = 100, overlap: int = 0) -> Optional[np.ndarray]:

        with self.lock:
            if len(self.buffer) < segment_size:
                return None

            segment_frames = []
            for _ in range(segment_size - overlap):
                if self.buffer:
                    segment_frames.append(self.buffer.popleft())
            if overlap > 0 and len(self.buffer) >= overlap:
                overlap_frames = list( self.buffer)[:overlap]
                segment_frames.extend(overlap_frames)

            if len(segment_frames) == segment_size:
                return np.array(segment_frames)
            else:
                for frame in reversed(segment_frames[:segment_size - overlap]):
                    self.buffer.appendleft(frame)
                return None

    def get_buffer_info(self) -> Dict[str,Any]:
        with self.lock:
            return {
                'current_frames': len(self.buffer),
                'max_frames': self.max_frames,
                'buffer_full_percentage': (len(self.buffer) / self.max_frames) * 100,
                'total_frames_added': self.total_frames_added,
                'frames_dropped': self.frames_dropped,
                'buffer_duration_seconds': len(self.buffer) / self.fps if self.fps > 0 else 0
            }

    def clear(self):
        with self.lock:
            self.buffer.clear()


class VideoStreamProcessor:
    """
    Processes continuous video streams from various sources (camera, file, RTSP, etc.).
    """

    def __init__(self,
                 source: Union[str, int] = 0,  # 0 for webcam, string for file/RTSP
                 fps: int = 30,
                 frame_size: Tuple[int,int] = (224,224)):

        self.source = source 
        self.fps = fps
        self.frame_size = frame_size
        self.is_file=isinstance(source, str) and os.path.isfile(source)


        self.cap=None
        self.is_running = False
        self.capture_thread = None

        self.frame_buffer = FrameBuffer(
            buffer_duration=30,
            fps=fps,
            max_frame_size = frame_size
        ) 

    def start_capture( self) -> bool:

        try:
            self.cap = cv2.VideoCapture(self.source) 

            if not self.cap.isOpened():
                print(f"Error: Could not open video source: {self.source}")
                return False 

            if not self.is_file:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"Video source opened: {self.source}")
            print(f"Actual FPS: {actual_fps}, Resolution: {actual_width}x{actual_height}")

            self.is_running = True
            self.capture_thread = threading.Thread( target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            return True

        except Exception as e:
            print(f"Error starting video capture: {e}")
            return False

    def _capture_loop(self):
        source_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
        frame_interval = 1.0 / source_fps if source_fps > 0 else 1.0 / self.fps
        next_frame_time = time.time()

        #frames here would change the temporal sampling
        while self.is_running and self.cap and self.cap.isOpened( ):
            ret, frame=self.cap.read()

            if not ret:
                print("End of video stream or capture error")
                break 

            self.frame_buffer.add_frame(frame)
            if self.is_file:
                next_frame_time += frame_interval
                time.sleep(max(0.0, next_frame_time - time.time()))

    def get_segment(self, segment_size: int=100) -> Optional[np.ndarray]:
        return self.frame_buffer.get_segment(segment_size)

    def pop_segment(self, segment_size: int = 100, overlap: int = 10) -> Optional[np.ndarray]:
        return self.frame_buffer.pop_segment(segment_size,overlap)

    def get_buffer_info(self) -> Dict[str, Any]:
        return self.frame_buffer.get_buffer_info()

    def stop_capture(self):
        self.is_running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        print("Video capture stopped")



class SegmentProcessor:
    """Turns a segment of buffered BGR frames into model input sampled exactly like the training clips."""

    def __init__(self, embedder):
        self.embedder = embedder 
        print(f"SegmentProcessor initialized: {embedder.max_frames} frames, every {embedder.frame_skip}th frame")

    def preprocess_segment(self,frames: np.ndarray) -> np.ndarray:

        sampled_frames = frames[::self.embedder.frame_skip][:self.embedder.max_frames]
        # Same resize, BGR -> RGB conversion and scaling as the training embeddings
        processed_frames = np.array([self.embedder.preprocess_frame(frame) for frame in sampled_frames])
        return self.embedder.pad_frames(processed_frames, self.embedder.max_frames)


class EmbeddingGenerator:
    def __init__(self,
                 model_type: str = 'vit',
                 embedder_path: str = None,
                 device: torch.device = None):

        self.model_type = model_type.lower()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model_type not in EMBEDDER_CLASSES:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        if not embedder_path:
            raise ValueError("embedder_path is required: the embedder must be the one that generated the "
                             "training embeddings")

        self.model = EMBEDDER_CLASSES[self.model_type].from_checkpoint(embedder_path, self.device)
        self.embedding_dim = self.model.output_dim
        print(f"EmbeddingGenerator initialized with {model_type} model on {self.device}")

    def generate_embeddings(self, frames: np.ndarray) -> np.ndarray:
        return self.model.extract_embeddings(frames)


class RealTimeVADEngine:

    def __init__(self,
                 video_source: Union[str,int] = 0,
                 embedding_model: str = 'vit',
                 sru_model_type: str = 'srupp',
                 embedder_path: Optional[str] = None,
                 cluster_weight: float = 0.3,
                 cluster_temperature: float=0.1,
                 segment_size: Optional[int] = None,
                 segment_overlap: int=10,
                 confidence_threshold: float=0.5, 
                 device: torch.device = None):
        """
        Initialize the Real-Time VAD Engine.

        Args:
            video_source: Video source (camera index, file path, or RTMP URL)
            embedding_model: Embedding model type ('cnn', 'resnet', 'vit')
            sru_model_type: SRU model type ('sru', 'srupp')
            embedder_path: Embedder checkpoint saved with the training embeddings
            cluster_weight: Weight for cluster-based enhancement (0.0-1.0)
            cluster_temperature: Softmax temperature applied to the cosine similarities
            segment_size: Number of buffered frames per segment (None: max_frames * frame_skip of the embedder,
                          i.e. the same span as a training clip)
            segment_overlap: Overlap between consecutive segments
            confidence_threshold: Minimum confidence for anomaly detection
            device: PyTorch device
        """
        self.video_source = video_source
        self.embedding_model_type = embedding_model
        self.sru_model_type = sru_model_type
        self.embedder_path = embedder_path 
        self.cluster_weight = cluster_weight
        self.cluster_temperature = cluster_temperature
        self.segment_size = segment_size
        self.segment_overlap=segment_overlap
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.stream_processor = None
        self.segment_processor = None
        self.embedding_generator = None
        self.sru_model = None
        self.cluster_model = None

        self.is_processing = False
        self.processing_thread = None
        self.results_queue = Queue()


        self.total_segments_processed = 0
        self.anomalies_detected=0
        self.processing_times = deque(maxlen=1000)  #Recent segments only, so long runs do not grow without bound

        self.category_names = get_category_names(12)
        print(f"RealTimeVADEngine initialized:")
        print(f"  - Video source: {video_source}")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - SRU model: {sru_model_type}")
        print(f"  - Device: {self.device}")
    def initialize_components(self): 
        print("Initializing components...")

        self.embedding_generator = EmbeddingGenerator( 
            model_type = self.embedding_model_type,
            embedder_path=self.embedder_path,
            device=self.device
        )
        embedder=self.embedding_generator.model
        clip_frames = embedder.max_frames * embedder.frame_skip
        if self.segment_size is None:
            self.segment_size = clip_frames 
        elif self.segment_size != clip_frames:
            print(f"Warning: segment_size={self.segment_size} differs from the {clip_frames} frames "
                  f"(max_frames * frame_skip) each training clip covered")
        if not 0 <= self.segment_overlap < self.segment_size:
            raise ValueError(f"segment_overlap must be between 0 and segment_size - 1, got {self.segment_overlap}")
        print(f"  - Segment size: {self.segment_size} frames")

        self.stream_processor = VideoStreamProcessor(
            source=self.video_source,
            fps = 30,
            frame_size=embedder.frame_size
        )

        self.segment_processor = SegmentProcessor(embedder)


        print("Components initialized successfully")

    def load_models(self,
                   sru_model_path: str,
                   cluster_model_path: Optional[str] = None):

        print("Loading models...")

        try:
            model_class = SRUModel if self.sru_model_type.lower() == 'sru' else SRUppModel
            # The architecture (input size, hidden size, classes, ...) is read from the checkpoint
            self.sru_model = model_class.from_checkpoint(sru_model_path, self.device)
            self.category_names = get_category_names(self.sru_model.num_classes)

            if self.embedding_generator.embedding_dim != self.sru_model.input_size:
                raise ValueError(f"The embedding model produces {self.embedding_generator.embedding_dim}-dim "
                                 f"embeddings but the SRU model expects {self.sru_model.input_size}")

            print(f"SRU model loaded from {sru_model_path}")

            if cluster_model_path and os.path.exists(cluster_model_path):
                cluster_model = VideoEmbeddingClusterer()
                cluster_model.load_model(cluster_model_path)

                if not cluster_model.class_centers:
                    print("Warning: No class centers found in cluster model")
                elif len(cluster_model.category_names) != self.sru_model.num_classes:
                    print(f"Warning: cluster model has {len(cluster_model.category_names)} classes but the SRU "
                          f"model has {self.sru_model.num_classes}. Skipping cluster enhancement.")
                else:
                    self.cluster_model=cluster_model
                    print(f"Clustering model loaded from {cluster_model_path}")
        except Exception as e: 
            print(f"Error loading models: {e}")
            raise

    def start_processing(self):
        """Start real-time video processing."""
        if not self.stream_processor or not self.sru_model:
            raise ValueError("Components not initialized or models not loaded")

        print("Starting real-time processing...")

        if not self.stream_processor.start_capture():
            raise RuntimeError("Failed to start video capture")
        self.is_processing = True
        self.processing_thread = threading.Thread( target=self._processing_loop,daemon=True)
        self.processing_thread.start()

        print("Real-time processing started")

    def _processing_loop(self): 
        print("Processing loop started...")
        while self.is_processing:
            try:
                segment = self.stream_processor.pop_segment(
                    segment_size=self.segment_size,
                    overlap=self.segment_overlap
                )

                if segment is None:
                    time.sleep(0.1)  #Wait for more frames
                    continue

                start_time = time.time()
                result=self._process_segment(segment)
                processing_time = time.time() - start_time

                if result:
                    result['processing_time'] = processing_time
                    result['timestamp'] = time.time()
                    self.results_queue.put(result)

                    self.total_segments_processed += 1
                    self.processing_times.append(processing_time)


                    if result['is_anomaly']:
                        self.anomalies_detected += 1
                        print(f"ANOMALY DETECTED: {result['predicted_category']} "
                              f"(Confidence: {result['confidence']:.4f})")
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)

    def _process_segment(self, segment: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            frames = self.segment_processor.preprocess_segment(segment)


            embeddings = self.embedding_generator.generate_embeddings(frames)


            # Prepared embeddings for SRU (add batch dimension and convert to tensor)
            embeddings_tensor = torch.from_numpy(embeddings).unsqueeze(0).to(self.device)

            embeddings_tensor = embeddings_tensor.transpose(0, 1)


            with torch.no_grad(): 
                sru_output = self.sru_model(embeddings_tensor)
                sru_probabilities = F.softmax(sru_output, dim=1).cpu().numpy()[0] 

            final_probabilities = sru_probabilities.copy()
            cluster_weights = None 
            enhancement_applied = False 

            if self.cluster_model is not None: 
                try:
                    similarities = self.cluster_model.class_similarities(embeddings,len(self.category_names))
                    cluster_weights = similarity_weights(similarities, self.cluster_temperature)

                    final_probabilities = (1 - self.cluster_weight) * sru_probabilities + \
                                         self.cluster_weight * cluster_weights
                    final_probabilities = final_probabilities / np.sum(final_probabilities)
                    enhancement_applied = True

                except Exception as e:
                    print(f"Warning: Cluster enhancement failed: {e}")
                    final_probabilities = sru_probabilities
                    cluster_weights = None 

            predicted_class=int(np.argmax(final_probabilities))
            confidence = np.max(final_probabilities)
            predicted_category = self.category_names[predicted_class]

            is_anomaly = predicted_class != 0 and confidence >= self.confidence_threshold
            anomaly_confidence = 1.0 - final_probabilities[0]

            result = {
                'predicted_class': predicted_class,
                'predicted_category': predicted_category,
                'confidence': float(confidence),
                'is_anomaly': bool(is_anomaly),
                'anomaly_confidence': float(anomaly_confidence),
                'sru_probabilities': sru_probabilities.tolist(),
                'final_probabilities': final_probabilities.tolist(),
                'cluster_weights': cluster_weights.tolist() if cluster_weights is not None else None,
                'enhancement_applied': enhancement_applied,
                'segment_size': len(segment)
            }



            return result 

        except Exception as e:
            print(f"Error processing segment: {e}")
            return None

    def get_latest_results(self,count: int=1) -> List[Dict[str,Any]]:
        results = []
        try:
            for _ in range( min(count, self.results_queue.qsize())):
                result = self.results_queue.get_nowait()
                results.append(result)
        except Empty:
            pass

        return results

    def get_statistics(self) -> Dict[str, Any]:
        buffer_info = self.stream_processor.get_buffer_info() if self.stream_processor else {}

        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

        return {
            'total_segments_processed': self.total_segments_processed,
            'anomalies_detected': self.anomalies_detected,
            'anomaly_rate': self.anomalies_detected / max(1, self.total_segments_processed),
            'average_processing_time': avg_processing_time,
            'pending_results': self.results_queue.qsize(),
            'buffer_info': buffer_info
        }
    def stop_processing(self):
        """Stop real-time processing."""
        print("Stopping real-time processing...")
        self.is_processing = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

        if self.stream_processor:
            self.stream_processor.stop_capture()

        print("Real-time processing stopped")


def demo_real_time_vad():
    """Demonstration of real-time VAD processing."""
    print("=== Real-Time Video Anomaly Detection Demo ===")


    engine = RealTimeVADEngine(
        video_source=0, 
        embedding_model='vit',
        sru_model_type='srupp',
        embedder_path='path/to/vit_embedder.pth',
        cluster_weight = 0.3,
        confidence_threshold=0.5 
    )


    try:
        #Initialize components (requires the embedder checkpoint saved with the training embeddings)
        # engine.initialize_components()

        # Load models (you need to provide actual model paths)
        #sru_model_path='path/to/best_srupp_model.pth',
        #)


        print("Note: Skipping model loading for demo (models not provided)")
        #Start processing
        # engine.start_processing()
        #for i in range(30):  # Run for 10 iterations
        #time.sleep(1)
        #
        #     # Get latest results
        #print(f"Result: {result['predicted_category']} " 
        #
        #     # Print statistics
        #     if i % 10 == 0:

        #Stop processing
        # engine.stop_processing()
        print("Demo completed")

    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    demo_real_time_vad()
