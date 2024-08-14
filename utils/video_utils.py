import cv2 
import subprocess

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path):
    # Enregistrer la vidéo avec le codec XVID
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

    # Chemin vers la vidéo de sortie après la conversion
    output_video_h264 = output_video_path.replace('.mp4', '_h264.mp4')

    # Commande ffmpeg pour convertir en H.264
    ffmpeg_cmd = f'ffmpeg -i {output_video_path} -vcodec libx264 {output_video_h264}'

    # Exécuter la commande ffmpeg
    try:
        subprocess.run(ffmpeg_cmd, check=True, shell=True)
        print(f"Video converted to H.264: {output_video_h264}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
