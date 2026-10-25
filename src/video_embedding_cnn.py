"""
CNN-based video embedding generation module.
This module provides video embedding generation using a custom CNN architecture.
It processes video frames and generates embeddings for video anomaly detection tasks.
"""

import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import glob 
from typing import List, Tuple, Optional
from multiprocessing import Pool
import gc


# Edit these values before running this file.
DATA_PATH = "/path/to/video/dataset"
OUTPUT_DIR = "/path/to/output/embeddings"
DATASET_TYPE = "binary"
MAX_FRAMES = 100
FRAME_SKIP=3
OUTPUT_DIM = 1024
SEED = 42
NUM_PROCESSES = 1 

BINARY_CATEGORY_LABELS = {
    'nonanomalous': 0, 'normal': 0, 'nonfight': 0,
    'anomalous': 1, 'fight': 1,
}
MULTICLASS_CATEGORY_LABELS = {
    'normal': 0, 'abuse': 1, 'arson': 2, 'assault': 3,
    'roadaccident': 4, 'burglary': 5, 'explosion': 6,
    'fighting': 7, 'robbery': 8, 'shooting': 9,
    'stealing': 10, 'vandalism': 11
}



class CNN(nn.Module):

    def __init__(self, output_dim: int = 1024): 

        super( CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        #For 224x224 input: ((224-2)/2-2)/2-2)/2 = 26
        self.fc1 = nn.Linear(128 * 26 * 26, output_dim)
        self.output_dim = output_dim

    def forward(self, x):

        x = F.relu(self.conv1(x)) 
        x = self.pool(x)
        x=F.relu(self.conv2(x))
        x = self.pool(x)
        x=F.relu(self.conv3(x))
        x = self.pool(x)


        x = x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))

        return x


class CNNVideoEmbedder:


    def __init__(self,
                 device: torch.device,
                 output_dim: int=1024,
                 frame_size: Tuple[int, int]=(224, 224),
                 max_frames: int=100,
                 frame_skip: int = 3, 
                 seed: int = 42):

        self.device = device
        self.output_dim=output_dim 
        self.frame_size = tuple(frame_size)
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self._warned_truncation = False

        self.config = {
            'output_dim': output_dim,
            'frame_size': self.frame_size,
            'max_frames': max_frames,
            'frame_skip': frame_skip,
            'seed': seed,
        }

        torch.manual_seed(seed)
        self.model = CNN(output_dim).to(device)
        self.model.eval()

        print(f"CNN Video Embedder initialized with output dim: {output_dim}")
        print(f"Device: {device}")
    def save_checkpoint(self, checkpoint_path: str) -> None:
        torch.save({'model_state_dict': self.model.state_dict(), 'config': self.config}, checkpoint_path)
        print(f"Embedder weights saved to: {checkpoint_path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device) -> 'CNNVideoEmbedder':
        """Rebuild the exact embedder (weights and frame sampling) that generated the training embeddings."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        embedder = cls(device=device, **checkpoint['config'])
        embedder.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Embedder weights loaded from: {checkpoint_path}")
        return embedder


    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a BGR frame and convert it to RGB in [0, 1]."""
        frame = cv2.resize(frame,self.frame_size)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return frame.astype( np.float32) / 255.0

    def get_frames(self, video_path: str,
                   max_frames: Optional[int] = None,
                   frame_skip: Optional[int]=None) -> np.ndarray:

        max_frames=self.max_frames if max_frames is None else max_frames
        frame_skip = self.frame_skip if frame_skip is None else frame_skip

        video = cv2.VideoCapture(video_path)
        if not video.isOpened( ):
            raise IOError(f"Error reading video file: {video_path}")

        frames = []
        frame_index = 0
        while not max_frames or len(frames) < max_frames:
            ret, frame = video.read()
            if not ret:
                break
            if frame_index % frame_skip == 0:
                frames.append(self.preprocess_frame(frame))
            frame_index += 1

        truncated = bool(max_frames) and len(frames) == max_frames and all(video.grab() for _ in range(frame_skip))
        video.release()
        if not frames:
            raise ValueError(f"No frames could be read from video: {video_path}")

        if truncated and not self._warned_truncation:
            print(f"Warning: {video_path} is longer than max_frames * frame_skip = {max_frames * frame_skip} frames; "
                  f"the rest of the video is ignored. Increase FRAME_SKIP to cover whole clips.")
            self._warned_truncation=True

        return np.array(frames)
    def pad_frames(self,frames: np.ndarray,target_frames: int) -> np.ndarray:

        current_frames = frames.shape[0]
        if current_frames == target_frames:
            return frames
        elif current_frames > target_frames:
            return frames[:target_frames]
        else:
            pad_shape = (target_frames - current_frames,) + frames.shape[1:]
            padding = np.zeros(pad_shape, dtype=frames.dtype)
            return np.concatenate([frames, padding], axis=0)

    def extract_embeddings(self,frames: np.ndarray,batch_size: int = 8) -> np.ndarray: 

        embeddings=[]
        with torch.no_grad():
            for start in range(0, len(frames), batch_size):
                frame_tensor = torch.from_numpy(frames[start:start + batch_size]).permute(0, 3, 1, 2).to(self.device)
                embedding = self.model(frame_tensor) 
                embeddings.append(embedding.cpu().numpy())

        return np.concatenate(embeddings,axis = 0)

    def process_video(self,
                     video_path: str,
                     max_frames: Optional[int]=None,
                     frame_skip: Optional[int] = None) -> np.ndarray:

        max_frames=self.max_frames if max_frames is None else max_frames
        frames = self.get_frames(video_path, max_frames, frame_skip)
        frames = self.pad_frames(frames, max_frames)
        return self.extract_embeddings(frames)

def process_single_video_cnn(args: Tuple[str, CNNVideoEmbedder, int, int]) -> Tuple[str, Optional[np.ndarray]]:

    video_path, embedder, max_frames, frame_skip=args
    try:
        embeddings = embedder.process_video(video_path, max_frames, frame_skip)
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        embeddings=None
    return video_path,embeddings


def generate_cnn_embeddings( video_paths: List[str],
                           labels: List[int],
                           device: torch.device,
                           output_dir: str,
                           max_frames: int = 100,
                           frame_skip: int = 3,
                           output_dim: int=1024,
                           seed: int = 42,
                           num_processes: int = 1) -> None:

    os.makedirs(output_dir, exist_ok=True)
    embedder = CNNVideoEmbedder(device,output_dim,max_frames=max_frames,frame_skip=frame_skip, seed=seed)

    print(f"Processing {len(video_paths)} videos...")
    print(f"Max frames per video: {max_frames}")
    print(f"Frame skip: {frame_skip}")
    print(f"Output dimension: {output_dim}")

    all_embeddings = []
    valid_labels = []
    valid_paths = []

    if num_processes > 1:
        print(f"Using multiprocessing with {num_processes} processes...")
        args_list = [(path,embedder,max_frames,frame_skip) for path in video_paths]

        with Pool(processes=num_processes) as pool:
            results=list(tqdm(
                pool.imap(process_single_video_cnn, args_list),
                total=len(args_list),
                desc="Extracting embeddings"
            ))

        for i, (video_path, embeddings) in enumerate(results):
            if embeddings is not None: 
                all_embeddings.append(embeddings)
                valid_labels.append(labels[i])
                valid_paths.append(video_path)
            else: 
                print(f"Skipping video {video_path} due to processing error")

    else:
        # Sequential processing
        for video_path, label in tqdm(zip(video_paths, labels),
                                     total=len(video_paths),
                                     desc="Extracting embeddings"):
            try:
                embeddings = embedder.process_video(video_path, max_frames, frame_skip)
                all_embeddings.append(embeddings)
                valid_labels.append(label)
                valid_paths.append(video_path)
            except Exception as e:
                print(f"Skipping video {video_path} due to error: {str(e)}")
                continue

    all_embeddings = np.array(all_embeddings)
    valid_labels = np.array(valid_labels)

    print(f"Generated embeddings shape: {all_embeddings.shape}")
    print(f"Generated labels shape: {valid_labels.shape}")

    embeddings_path = os.path.join(output_dir, 'cnn_embeddings.npy')
    labels_path = os.path.join(output_dir, 'cnn_labels.npy')
    paths_path = os.path.join(output_dir, 'cnn_paths.npy')

    np.save(embeddings_path, all_embeddings)
    np.save(labels_path, valid_labels) 
    np.save(paths_path, np.array(valid_paths))
    embedder.save_checkpoint(os.path.join(output_dir, 'cnn_embedder.pth'))

    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Labels saved to: {labels_path}")
    print(f"Video paths saved to: {paths_path}")

    del embedder
    gc.collect()

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device selected: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device selected: CUDA")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")
    return device



def load_video_dataset(data_path: str, dataset_type: str = 'binary') -> Tuple[List[str], List[int]]:

    if dataset_type == 'binary':
        category_map = BINARY_CATEGORY_LABELS
    elif dataset_type == 'multiclass':
        # Multiclass classification
        category_map=MULTICLASS_CATEGORY_LABELS
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    video_paths = []
    labels = []
    skipped_categories = set()

    pattern = os.path.join(data_path, '*', '*.mp4')
    for path in sorted(glob.glob(pattern)):
        category=os.path.basename(os.path.dirname(path)).lower()
        if category not in category_map:
            skipped_categories.add(category)
            continue
        video_paths.append(path)
        labels.append(category_map[category])

    if skipped_categories:
        print(f"Warning: skipped videos in unknown folders: {sorted(skipped_categories)}")


    return video_paths, labels


def main():
    device=setup_device()

    print("Loading dataset...")
    video_paths, labels=load_video_dataset(DATA_PATH, DATASET_TYPE)
    print(f"Found {len(video_paths)} videos")

    generate_cnn_embeddings(
        video_paths=video_paths,
        labels = labels, 
        device=device,
        output_dir = OUTPUT_DIR,
        max_frames=MAX_FRAMES,
        frame_skip=FRAME_SKIP,
        output_dim = OUTPUT_DIM,
        seed=SEED,
        num_processes=NUM_PROCESSES
    )

    print("CNN embedding generation completed!")


if __name__ == "__main__":
    main()
