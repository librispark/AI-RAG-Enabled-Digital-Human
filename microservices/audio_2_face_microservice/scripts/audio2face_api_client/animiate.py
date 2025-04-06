import os
import csv
import time
import pygame
import cv2
import numpy as np

# Initialize Pygame
pygame.init()


# Function to load CSV files
def load_csv(file_path):
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    keyframes = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keyframes.append(row)
    return keyframes


# Load keyframes for animation and emotions
animation_keyframes = load_csv('./20240922_203602_800931/animation_frames.csv')
emotion_output = load_csv('./20240922_203602_800931/a2f_smoothed_emotion_output.csv')


# Load the WAV file for audio playback
def play_audio(file):
    if not os.path.exists(file):
        print(f"Audio file not found: {file}")
        return
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()


# Function to scale or move parts of the image (like mouth, eyes)
def apply_blendshape(image, base_image, blendshape_data):
    """
    Apply blendshape values to the 2D face image.
    For example, scale the mouth to simulate a smile, or move the eyebrows.
    """
    # Updated regions for facial features (based on the user's values)
    mouth_region = (280, 325, 479, 572)  # y1, y2, x1, x2
    left_eye_region = (185, 219, 447, 505)
    right_eye_region = (182, 216, 547, 604)

    # Extract blendshape values from the keyframe
    mouth_smile_left = float(blendshape_data['blendShapes.MouthSmileLeft'])
    mouth_smile_right = float(blendshape_data['blendShapes.MouthSmileRight'])

    # Extract the original mouth region from the base image (not the current frame)
    mouth_y1, mouth_y2, mouth_x1, mouth_x2 = mouth_region
    original_mouth = base_image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

    # Scale the mouth based on smile blendshape values
    scale_x = 1 + mouth_smile_left * 0.5  # Adjust scaling factor as needed
    scaled_width = int(original_mouth.shape[1] * scale_x)
    scaled_mouth = cv2.resize(original_mouth, (scaled_width, original_mouth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Adjust the scaled mouth to fit back into the original region size
    if scaled_mouth.shape[1] > original_mouth.shape[1]:
        # Crop if scaled mouth is too wide
        scaled_mouth = scaled_mouth[:, :original_mouth.shape[1]]
    elif scaled_mouth.shape[1] < original_mouth.shape[1]:
        # Pad if scaled mouth is too narrow
        padding = original_mouth.shape[1] - scaled_mouth.shape[1]
        scaled_mouth = np.pad(scaled_mouth, ((0, 0), (0, padding), (0, 0)), mode='constant')

    # Place the resized mouth back into the original image for this frame
    image[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = scaled_mouth

    return image


# Incorporate emotion data into the animation
def apply_emotion(image, emotion_data):
    """
    Modify the animation based on emotion data.
    """
    joy = float(emotion_data['emotion_values.joy'])
    grief = float(emotion_data['emotion_values.grief'])

    # For example, increase the smile intensity if joy is high
    if joy > 0.5:
        mouth_region = (280, 325, 479, 572)  # Updated mouth region
        mouth_y1, mouth_y2, mouth_x1, mouth_x2 = mouth_region
        mouth = image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

        # Scale the smile according to joy level
        scale_x = 1 + joy * 0.5
        scaled_mouth = cv2.resize(mouth, None, fx=scale_x, fy=1, interpolation=cv2.INTER_LINEAR)
        image[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = scaled_mouth

    # You can also apply sadness by inverting the smile, lowering the mouth corners
    if grief > 0.5:
        mouth_region = (280, 325, 479, 572)  # Updated mouth region
        mouth_y1, mouth_y2, mouth_x1, mouth_x2 = mouth_region
        mouth = image[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

        # Invert scaling to simulate a frown
        scale_x = 1 - grief * 0.5
        scaled_mouth = cv2.resize(mouth, None, fx=scale_x, fy=1, interpolation=cv2.INTER_LINEAR)
        image[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = scaled_mouth

    return image


# Animate based on the keyframes and emotions
def animate_2d_image_with_emotions(image_path, animation_keyframes, emotion_output, audio_file):
    # Load the base face image
    base_image = cv2.imread(image_path)
    image = base_image.copy()  # Make a working copy of the image

    # Set up Pygame window for displaying animation
    screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
    pygame.display.set_caption('2D Animation with Emotions and Audio Sync')

    # Play the audio file
    play_audio(audio_file)

    start_time = time.time()

    # Loop through keyframes
    for i, frame in enumerate(animation_keyframes):
        # Get current time and wait for the keyframe time
        current_time = time.time() - start_time
        while current_time < float(frame['timeCode']):
            current_time = time.time() - start_time

        # Apply blendshape data to the image, starting from the original base image each time
        blendshape_frame = animation_keyframes[i]
        image = apply_blendshape(image.copy(), base_image, blendshape_frame)

        # Apply emotion data if available
        emotion_frame = emotion_output[i]  # Assuming same length keyframes for simplicity
        image = apply_emotion(image.copy(), emotion_frame)

        # Convert image to Pygame surface and display
        img_surface = pygame.surfarray.make_surface(cv2.transpose(image))
        screen.blit(img_surface, (0, 0))
        pygame.display.update()

        # Check for quit events (optional)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                return

        # Wait for the next frame
        pygame.time.wait(100)  # Adjust for smoother animation

    # Wait for the audio to finish before exiting
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# Example usage
image_path = './person_base.png'  # Your 2D face image in the same directory
csv_file = './20240922_203602_800931/animation_frames.csv'  # Animation keyframes
emotion_file = './20240922_203602_800931/a2f_smoothed_emotion_output.csv'  # Smoothed emotion data
audio_file = './20240922_203602_800931/out.wav'  # Audio file

# Load CSV files for animation and emotions
animation_keyframes = load_csv(csv_file)
emotion_output = load_csv(emotion_file)

# Run the animation with emotions and audio sync
animate_2d_image_with_emotions(image_path, animation_keyframes, emotion_output, audio_file)

# Clean up Pygame
pygame.quit()
