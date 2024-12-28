import cv2
import os

def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """

    # Open the video file using VideoCapture
    cap = cv2.VideoCapture(videopath)
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        print('Error: Could not open the video file')
        return None, None, None, None

    # Get the size of frames (width and height) and the frame rate
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the video frames
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the video frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    
    # List to store the frames
    frames = []
    
    # Read the frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Release the video capture object
    cap.release()

    return frames, frame_width, frame_height, frame_rate

# Path to the video file
videopath = 'colors/incorrect.mp4'

# Read the video and obtain the frames and properties
frames, frame_width, frame_height, frame_rate = read_video(videopath)

# Print video properties for verification
if frames:
    print(f"Video loaded successfully: {len(frames)} frames")
    print(f"Frame width: {frame_width}, Frame height: {frame_height}, Frame rate: {frame_rate}")
else:
    print("Failed to load video.")

# Create a directory to save frames with the name of the video file
video_name = os.path.splitext(os.path.basename(videopath))[0]
output_dir = f"fotos/frames/{video_name}"
os.makedirs(output_dir, exist_ok=True)

reference_frame = None

# Step 1: Show frames and select the reference frame
for i, frame in enumerate(frames):
    # Show the frame
    cv2.imshow('Video', frame)
    
    # Wait for key press
    key = cv2.waitKey(0)  # 0 means wait indefinitely for a key press
    
    # If 'n' is pressed, continue to the next frame
    if key == ord('n'):
        continue
    
    # If 's' is pressed, select the current frame as the reference frame
    elif key == ord('s'):
        reference_frame = frame.copy()  # Copy the current frame as the reference
        frame_filename = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(frame_filename, reference_frame)  # Save the selected frame
        print(f"Frame {i} saved as {frame_filename}")
    
    # If 'q' is pressed, quit the selection loop
    elif key == ord('q'):
        print("Exiting frame selection.")
        break

cv2.destroyAllWindows()

# Step 2: Compute the difference between the reference frame and the rest of the frames
if reference_frame is not None:
    reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    for i, frame in enumerate(frames):
        # Convert the current frame to grayscale
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the absolute difference between the reference and the current frame
        diff = cv2.absdiff(reference_frame_gray, current_frame_gray)
        
        # Show the difference
        cv2.imshow('Difference', diff)
        
        # Wait for a key press to continue
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to quit
            print("Exiting difference visualization.")
            break

cv2.destroyAllWindows()