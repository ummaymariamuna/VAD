"""
Reference configuration for the real-time video anomaly detection system.

Not imported by the scripts; the values mirror their defaults. The SRU/SRU++ architecture and the frame sampling are
read from the saved checkpoints, so they are not configured here.
"""
VIDEO_CONFIG = {
    'default_fps': 30,
    'frame_size': (224, 224),  # (width, height)
    'buffer_duration': 30,
    'segment_size': None,       # frames per segment (None: max_frames * frame_skip of the embedder)
    'segment_overlap': 10,
}

MODEL_CONFIG={
    'embedding_models': {
        'cnn': {
            'output_dim': 1024,
            'max_frames': 100,
            'frame_skip': 3,
        }, 
        'resnet': {
            'output_dim': 1024,
            'use_attention': True,
            'max_frames': 150,
            'frame_skip': 2,
        },
        'vit': {
            'output_dim': 2048,
            'vit_model': 'vit_small_patch16_224',
            'use_pretrained': True,
            'max_frames': 75,
            'frame_skip': 4,
        }
    },
}
#Processing Configuration
PROCESSING_CONFIG={
    'confidence_threshold': 0.5,
    'cluster_weight': 0.3,
    'cluster_temperature': 0.1,
    'anomaly_threshold': 0.5,
    'max_queue_size': 1000,
}

# Category Labels 
CATEGORY_NAMES = [
    "Normal", "Abuse", "Arson", "Assault", "Road Accident", "Burglary",
    "Explosion", "Fighting", "Robbery", "Shooting", "Stealing", "Vandalism"
]

DEVICE_CONFIG={
    'prefer_gpu': True,
    'gpu_memory_fraction': 0.8,
}

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'log_file': 'real_time_vad.log',
}

# Output Configuration 
OUTPUT_CONFIG={
    'save_results': True,
    'results_format': 'json',
    'display_video': False,
    'statistics_interval': 10,  # seconds
}
