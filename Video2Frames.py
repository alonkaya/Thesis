from pytube import YouTube
import cv2
import os
import glob
import shutil

def download_video(url, path):
    yt = YouTube(url)
    ys = yt.streams.filter(file_extension='mp4').get_highest_resolution()
    ys.download(filename=path)
    return path

def extract_frames(video_path, timestamps_microsec, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    for i, timestamp_microsec in enumerate(timestamps_microsec):
        timestamp_ms = timestamp_microsec / 1000  # Convert from microseconds to milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, image = cap.read()
        if success:
            output_path = os.path.join(output_dir, f'{i:06}.jpg')
            cv2.imwrite(output_path, image)
            print(f'Frame for timestamp {timestamp_microsec} µs extracted and saved to {output_path}')
        else:
            print(f'Error: Could not extract frame for timestamp {timestamp_microsec} µs')

    cap.release()

def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    url = lines[0].strip()
    timestamps = [int(line.split()[0]) for line in lines[1:]]
    return url, timestamps

def process_files(directory_from, directory_to, limit):
    files = glob.glob(os.path.join(directory_from, '*.txt'))[10:limit]

    for file_path in files:
        try:
            print(f'Processing {file_path}')
            filename = os.path.splitext(os.path.basename(file_path))[0]
            if filename in os.listdir(directory_to): continue
            
            url, timestamps = parse_file(file_path)
            video_path = download_video(url, path=os.path.join(directory_from, f'{filename}.mp4'))

            output_dir = os.path.join(directory_to, filename)
            image_0_dir = os.path.join(output_dir, 'image_0')

            extract_frames(video_path, timestamps, image_0_dir)

            shutil.copy(file_path, output_dir)

            # Optionally, remove the downloaded video if not needed
            os.remove(video_path)
        except Exception as e:
            print(f'Error: {e}')
if __name__ == "__main__":
    # Specify the directory containing the text files
    # directory_from = 'RealEstate10K/train'
    # directory_to = 'RealEstate10K/train_images'
    # process_files(directory_from, directory_to, limit=120)

    directory_from = 'RealEstate10K/test'
    directory_to = 'RealEstate10K/val_images'
    process_files(directory_from, directory_to, limit=30)
