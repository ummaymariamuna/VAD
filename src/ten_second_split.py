"""
Video segmentation utility used to create ten-second clips.

This file is a Python script that segments a video into multiple videos of a specified duration.
It uses the OpenCV library to read and write video files, and the os and tqdm libraries for file and progress bar
operations, respectively.

The segment_video function takes an input video file, an output video file pattern, and an optional segment duration as
input. It opens the input video file using OpenCVs' VideoCapture class, and then calculates the frame rate and the
number of frames for the segment duration. It then reads the video frames sequentially until the stream ends (the frame
count reported by OpenCV is only an estimate, so it is not used to drive the loop). Each segment is written to a new
video file with a VideoWriter object, which is released once the segment is full or the video ends.

The main part of the script demonstrates how to use the segment_video function to segment all input videos in a
specified directory. It first specifies the input and output directories, and then loops over all input videos in the
input directory. For each video, it calls the segment_video function to segment the video into multiple videos of the
specified duration, and saves the segmented videos to the output directory.

However, Some videos may have less than 10 seconds of content, so the last segment may be shorter than the specified
duration. You may need to handle this case separately if necessary.
"""

import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool 


def segment_video( args):
    """
    This function segments a video into multiple videos of a specified duration.
    Arguments is a tuple containing input file, output file pattern, and segment duration.
    """
    in_file, out_file, segment_duration=args
    cap=cv2.VideoCapture(in_file) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        print(f"Skipping {in_file}: could not read the video or its frame rate")
        cap.release()
        return


    segment_frames = max(1, int(round(segment_duration * fps)))
    frame_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    segment_index = 0
    frames_in_segment=0
    out=None

    # Read frames until the stream ends, starting a new segment every segment_frames frames 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if out is None:
            output_file = out_file % segment_index
            # The VideoWriter object takes the output file name, codec, frame rate, and frame size as input
            out = cv2.VideoWriter(output_file, cv2.VideoWriter.fourcc(*'mp4v'), fps, frame_size)

        out.write(frame)
        frames_in_segment += 1

        if frames_in_segment == segment_frames: 
            out.release()
            out = None
            frames_in_segment=0
            segment_index += 1

    if out is not None:
        out.release() 

    cap.release()


if __name__ == '__main__':
    input_dir = '/path/to/folder'
    output_dir = '/path/to/output/folder'
    os.makedirs(output_dir,exist_ok=True) 

    args_list = []

    # Loop over all input videos in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_file=os.path.join(input_dir,filename)
            output_pattern = os.path.join(output_dir, os.path.splitext(filename)[0] + '_%03d.mp4')
            args_list.append((input_file, output_pattern, 10))  #Assuming default segment duration of 10 seconds

    # Started Pool for multiprocessing
    with Pool() as pool:
        list(tqdm(pool.imap(segment_video, args_list), total = len(args_list))) 
