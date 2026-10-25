"""
Video augmentation module for video anomaly detection (VAD).
This module provides comprehensive video augmentation functionality for video anomaly detection tasks.
It supports various augmentation techniques including:
- Horizontal and vertical flipping
- Brightness and contrast adjustment
- Gaussian noise addition
- Temporal augmentations
"""

import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from typing import Tuple,List, Optional, Dict, Any
import random



INPUT_DIR = "/path/to/input/videos"
OUTPUT_DIR = "/path/to/output/videos"
BRIGHTNESS_RANGE=(0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
NOISE_STD_RANGE = (0, 25)
FLIP_PROBABILITY = 0.5
AUGMENT_PROBABILITY=0.8
AUGMENTATIONS_PER_VIDEO = 3
NUM_PROCESSES = 4


class VideoAugmentor:

    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 contrast_range: Tuple[float,float]=(0.7,1.3),
                 noise_std_range: Tuple[float, float] = (0, 25),
                 flip_probability: float=0.5,
                 augment_probability: float = 0.8):

        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std_range=noise_std_range
        self.flip_probability=flip_probability
        self.augment_probability = augment_probability
    def adjust_brightness_contrast(self, frame: np.ndarray, 
                                 brightness: float, contrast: float) -> np.ndarray:
        
        frame_float=frame.astype(np.float32)
        
        # Applied brightness and contrast: new_pixel = contrast * pixel + brightness_offset
        brightness_offset = (brightness - 1.0) * 128
        adjusted = contrast * frame_float + brightness_offset        
        adjusted = np.clip(adjusted, 0, 255)

        return adjusted.astype(np.uint8) 

    def add_gaussian_noise(self, frame: np.ndarray, noise_std: float,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:

        if noise_std <= 0:
            return frame

        rng = rng if rng is not None else np.random.default_rng()
        noise = rng.normal(0, noise_std, frame.shape).astype(np.float32)
        noisy_frame = frame.astype(np.float32) + noise        
        noisy_frame = np.clip(noisy_frame, 0, 255)
        
        return noisy_frame.astype(np.uint8)

    def flip_frame(self,frame: np.ndarray, flip_type: str) -> np.ndarray:


    
        if flip_type == 'horizontal':
            return cv2.flip(frame, 1)  # Horizontal flip
        elif flip_type == 'vertical':
            return cv2.flip(frame,0)
        elif flip_type == 'both':
            return cv2.flip(frame,-1)  # Both horizontal and vertical
        else:
            return frame
    
    def augment_frame( self,frame: np.ndarray,augmentation_params: Dict[str, Any],
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:

        augmented_frame = frame.copy()
        #Applied brightness and contrast adjustment
        if 'brightness' in augmentation_params and 'contrast' in augmentation_params:
            augmented_frame=self.adjust_brightness_contrast(
                augmented_frame, 
                augmentation_params['brightness'], 
                augmentation_params['contrast']
            )
        
        if 'noise_std' in augmentation_params:
            augmented_frame=self.add_gaussian_noise(
                augmented_frame,
                augmentation_params['noise_std'],
                rng
            ) 

        
        if 'flip_type' in augmentation_params:
            augmented_frame = self.flip_frame(
                augmented_frame, 
                augmentation_params['flip_type']
            )
        
        return augmented_frame
    def generate_augmentation_params(self, force: bool = False) -> Dict[str, Any]:

        params={}

        # Randomly decide whether to apply augmentations (force is used for "_aug_" copies, which would otherwise be
        # exact duplicates of the "_original" copy)
        if not force and random.random() > self.augment_probability:
            return params  # Return empty dict for no augmentation

        params['noise_seed'] = random.randrange(2 ** 32)


        params['brightness'] = random.uniform(*self.brightness_range)
        params['contrast'] = random.uniform(*self.contrast_range)


        params['noise_std'] = random.uniform(*self.noise_std_range)
        
        if random.random() < self.flip_probability:
            flip_options = ['horizontal', 'vertical', 'both']
            params['flip_type'] = random.choice(flip_options)
        return params 
    def augment_video(self,input_path: str, output_path: str, 
                     augmentation_params: Optional[Dict[str,Any]] = None) -> bool:
      
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {input_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0:
                print(f"Error: Could not read frame rate of {input_path}")
                cap.release()
                return False


            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened(): 
                print(f"Error: Could not create output video {output_path}")
                cap.release()
                return False
            if augmentation_params is None:
                augmentation_params=self.generate_augmentation_params()

            rng = np.random.default_rng(augmentation_params.get('noise_seed'))
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if augmentation_params:
                    augmented_frame = self.augment_frame(frame, augmentation_params, rng)
                else:
                    augmented_frame = frame
                
                out.write(augmented_frame) 
                frame_count += 1


            cap.release()
            out.release( )

            return True
            
        except Exception as e:
            print(f"Error processing video {input_path}: {str(e)}")
            return False


def augment_single_video(args: Tuple[str,str, VideoAugmentor, Dict[str, Any]]) -> Tuple[str, bool]:

    input_path, output_path, augmentor, augmentation_params = args
    success = augmentor.augment_video(input_path, output_path, augmentation_params)
    return input_path, success




def create_augmented_dataset(input_dir: str, output_dir: str, 
                           augmentor: VideoAugmentor,
                           augmentations_per_video: int = 3,
                           num_processes: int = 4) -> None:

    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = ['.mp4']
    video_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} video files")
    print(f"Creating {augmentations_per_video} augmentations per video")

    args_list = []
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        
        original_output = os.path.join(output_dir, f"{base_name}_original.mp4")
        args_list.append((input_path, original_output, augmentor, {}))
        
        for i in range(augmentations_per_video):
            augmented_output = os.path.join(output_dir, f"{base_name}_aug_{i:03d}.mp4")
            args_list.append((input_path, augmented_output, augmentor, augmentor.generate_augmentation_params(force=True)))
    
    print(f"Processing {len(args_list)} videos using {num_processes} processes...")

    with Pool(processes = num_processes) as pool:
        results = list(tqdm(
            pool.imap(augment_single_video, args_list),
            total = len(args_list),
            desc="Augmenting videos"
        ))

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    print(f"\nAugmentation completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed videos:")
        for input_path, success in results: 
            if not success:
                print(f"  - {input_path}")


def main( ):
    augmentor = VideoAugmentor(
        brightness_range=BRIGHTNESS_RANGE,
        contrast_range=CONTRAST_RANGE,
        noise_std_range = NOISE_STD_RANGE,
        flip_probability=FLIP_PROBABILITY,
        augment_probability=AUGMENT_PROBABILITY
    )

    create_augmented_dataset(
        INPUT_DIR,
        OUTPUT_DIR, 
        augmentor,
        AUGMENTATIONS_PER_VIDEO,
        NUM_PROCESSES
    )


if __name__ == "__main__":
    main()
