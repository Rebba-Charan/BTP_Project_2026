import cv2

# Load your video file
video = cv2.VideoCapture("videos/dataset_20230713-133150_0_S02_T76_E2.mp4")

# Extract the FPS property
fps = video.get(cv2.CAP_PROP_FPS)

print(f"Frames per second (FPS): {fps}")

video.release()
