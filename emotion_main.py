import cv2
import numpy as np
import os
import tensorflow as tf

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
print("Loading model...")
model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'model_file_30epochs.h5'))
print(f"Model input shape: {model.input_shape}")
print("Model loaded successfully!")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(os.path.join(SCRIPT_DIR, 'haarcascade_frontalface_default.xml'))

# Define emotions in the EXACT order the model was trained (alphabetical = Keras default)
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Emoji / color per emotion for richer display
EMOTION_COLORS = {
    'angry':     (0,   0,   220),   # red
    'disgusted': (0,   128, 0),     # dark green
    'fearful':   (200, 0,   200),   # purple
    'happy':     (0,   220, 220),   # yellow-green
    'neutral':   (200, 200, 200),   # grey
    'sad':       (220, 100, 0),     # blue-ish
    'surprised': (0,   165, 255),   # orange
}

# Confidence threshold — predictions below this are shown as 'uncertain'
CONFIDENCE_THRESHOLD = 0.30

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to improve detection in varying light
    gray_eq = cv2.equalizeHist(gray)

    # Detect faces — lowered minNeighbors (3→5) and scaleFactor (1.1) for better recall
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # --- FIX 1: Extract grayscale ROI (no equalized — keep natural pixel values) ---
        face_roi = gray[y:y + h, x:x + w]

        try:
            # Resize to 48x48 as model expects
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Resize error: {e}")
            continue

        # --- FIX 2: Correct preprocessing ---
        # Convert to float32 explicitly, normalize to [0, 1]
        resized_face = resized_face.astype('float32') / 255.0

        # Reshape to (1, 48, 48, 1) — batch of 1, grayscale
        input_face = np.expand_dims(resized_face, axis=(0, -1))  # shape: (1, 48, 48, 1)

        try:
            # --- FIX 3: Suppress per-frame verbose predict output ---
            prediction = model.predict(input_face, verbose=0)[0]  # shape: (7,)
            emotion_index = int(np.argmax(prediction))
            confidence = float(prediction[emotion_index])
            emotion_label = emotions[emotion_index]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        # Use threshold to avoid low-confidence noise
        display_label = emotion_label if confidence >= CONFIDENCE_THRESHOLD else 'uncertain'
        color = EMOTION_COLORS.get(emotion_label, (255, 255, 255))

        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw label + confidence above the box
        label_text = f"{display_label} ({confidence*100:.0f}%)"
        label_y = y - 10 if y - 10 > 15 else y + h + 20
        cv2.putText(frame, label_text, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        # --- FIX 4: Draw probability bars for ALL emotions (diagnostic) ---
        bar_x = x + w + 10
        bar_max_w = 100
        bar_h = 12
        bar_spacing = 16
        panel_h = len(emotions) * bar_spacing + 10

        # Only draw bars if they fit in the frame
        if bar_x + bar_max_w + 50 < frame.shape[1]:
            # Background panel
            cv2.rectangle(frame,
                          (bar_x - 2, y),
                          (bar_x + bar_max_w + 52, y + panel_h),
                          (30, 30, 30), -1)
            for i, (em, prob) in enumerate(zip(emotions, prediction)):
                bar_y = y + i * bar_spacing + 5
                fill_w = int(prob * bar_max_w)
                em_color = EMOTION_COLORS.get(em, (200, 200, 200))
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 60), -1)
                # Filled bar
                if fill_w > 0:
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), em_color, -1)
                # Emotion name
                cv2.putText(frame, f"{em[:4]}", (bar_x + bar_max_w + 3, bar_y + bar_h - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, em_color, 1, cv2.LINE_AA)

    # Instructions overlay
    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # Display
    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
