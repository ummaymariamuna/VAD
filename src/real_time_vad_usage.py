"""Manually configured real-time video anomaly detection example."""

import os
import time
import json

from real_time_vad_processor import RealTimeVADEngine


VIDEO_SOURCE = 0
RTSP_URL=None
EMBEDDER_PATH = "/path/to/output/embeddings/vit_embedder.pth"
SRU_MODEL_PATH = "models/best_srupp_model.pth"
CLUSTER_MODEL_PATH = None
EMBEDDING_MODEL = "vit"
SRU_MODEL_TYPE = "srupp"
SEGMENT_SIZE = None
SEGMENT_OVERLAP = 10
CLUSTER_WEIGHT=0.3
CLUSTER_TEMPERATURE = 0.1
CONFIDENCE_THRESHOLD=0.5
OUTPUT_DIR = "real_time_results"
SAVE_RESULTS = False
DISPLAY_VIDEO = False
DURATION = 60
STATS_INTERVAL = 10


def main():
    video_source = RTSP_URL if RTSP_URL else VIDEO_SOURCE
    print("=== Real-Time Video Anomaly Detection ===")
    print(f"Video source: {video_source}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"SRU model: {SRU_MODEL_TYPE}")
    print(f"Segment size: {SEGMENT_SIZE or 'from embedder'} frames")
    print(f"Cluster weight: {CLUSTER_WEIGHT}")
    print("=" * 50)
    try:
        engine = RealTimeVADEngine( 
            video_source=video_source,
            embedding_model = EMBEDDING_MODEL,
            sru_model_type=SRU_MODEL_TYPE,
            embedder_path=EMBEDDER_PATH,
            cluster_weight=CLUSTER_WEIGHT,
            cluster_temperature = CLUSTER_TEMPERATURE,
            segment_size=SEGMENT_SIZE,
            segment_overlap=SEGMENT_OVERLAP,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        print("Initializing components...")
        engine.initialize_components()

        print("Loading models...")
        engine.load_models(
            sru_model_path = SRU_MODEL_PATH,
            cluster_model_path=CLUSTER_MODEL_PATH
        )

        if SAVE_RESULTS:
            os.makedirs(OUTPUT_DIR,exist_ok = True) 
            results_file = os.path.join(OUTPUT_DIR, f'real_time_results_{int(time.time())}.json')
            all_results = []

        print("Starting real-time processing...")
        engine.start_processing()
        
        print("Waiting for video buffer to fill...")
        time.sleep(5)
        
        start_time = time.time()
        last_stats_time = start_time
        
        try: 
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if DURATION > 0 and elapsed_time >= DURATION:
                    print(f"Reached duration limit of {DURATION} seconds")
                    break

                results=engine.get_latest_results(count=10)

                for result in results:
                    timestamp = result.get('timestamp', current_time)
                    category = result['predicted_category']
                    confidence = result['confidence']
                    is_anomaly = result['is_anomaly']
                    processing_time = result.get('processing_time', 0)

                    status = "ANOMALY" if is_anomaly else "Normal"
                    print(f"[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] "
                          f"{status} | {category} | Confidence: {confidence:.4f} | "
                          f"Process Time: {processing_time:.3f}s")
                    
                    if SAVE_RESULTS:
                        all_results.append(result)
                if current_time - last_stats_time >= STATS_INTERVAL:
                    stats = engine.get_statistics()
                    print(f"\n--- Statistics (Elapsed: {elapsed_time:.1f}s) ---")
                    print(f"Segments processed: {stats['total_segments_processed']}")
                    print(f"Anomalies detected: {stats['anomalies_detected']}")
                    print(f"Anomaly rate: {stats['anomaly_rate']:.3f}")
                    print(f"Avg processing time: {stats['average_processing_time']:.3f}s")
                    print(f"Buffer usage: {stats['buffer_info'].get('buffer_full_percentage', 0):.1f}%")
                    print(f"Pending results: {stats['pending_results']}")
                    print("-" * 50)
                    
                    last_stats_time = current_time
                time.sleep(0.1)

        except KeyboardInterrupt: 
            print("\nProcessing interrupted by user")

        print("Stopping processing...")
        engine.stop_processing()

        if SAVE_RESULTS and all_results:
            print(f"Saving {len(all_results)} results to {results_file}")
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        final_stats = engine.get_statistics()
        print(f"\n=== Final Statistics ===")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")
        print(f"Segments processed: {final_stats['total_segments_processed']}")
        print(f"Anomalies detected: {final_stats['anomalies_detected']}")
        print(f"Anomaly rate: {final_stats['anomaly_rate']:.3f}")
        print(f"Average processing time: {final_stats['average_processing_time']:.3f}s")

    except Exception as e:
        print(f"Error: {e}")
        raise

def test_with_sample_video():
    """Test the system with a sample video file."""
    print("=== Testing with Sample Video ===")


    sample_video = "test_video.mp4"  
    embedder_path = "embeddings/vit_embedder.pth"
    sru_model_path = "models/best_srupp_model.pth" 
    cluster_model_path = "clustering_results/clustering_model.pkl" 
    if not os.path.exists(sample_video):
        print(f"Sample video not found: {sample_video}")
        return
    
    if not os.path.exists(sru_model_path):
        print(f"SRU model not found: {sru_model_path}")
        return
    
    engine=RealTimeVADEngine(
        video_source=sample_video,
        embedding_model='vit',
        sru_model_type='srupp',
        embedder_path=embedder_path,
        confidence_threshold = 0.5
    )
    try: 
        engine.initialize_components()
        engine.load_models(sru_model_path,cluster_model_path)
        engine.start_processing()

        time.sleep(30)

        engine.stop_processing() 
        
        stats = engine.get_statistics()
        print(f"Test completed - Processed {stats['total_segments_processed']} segments")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()
