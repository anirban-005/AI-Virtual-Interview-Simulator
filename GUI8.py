import pyttsx3  # Text-to-speech engine for speaking questions
import sounddevice as sd  # Records audio from microphone
from scipy.io.wavfile import write  # Saves recorded audio as WAV file
import speech_recognition as sr  # Converts speech to text
import tempfile  # Creates temporary files for audio storage
import os  # File and path operations
import csv  # Logging interview data
import json  # Loading sample answers configuration
import time  # Performance optimization and delays
from sentence_transformers import SentenceTransformer, util  # AI model for semantic similarity
from textblob import TextBlob  # Grammar and sentiment analysis
import google.generativeai as genai  # Google's Gemini AI for feedback
import cv2  # OpenCV for webcam and image processing
import numpy as np  # Numerical operations for image arrays
import tensorflow as tf  # Deep learning for emotion detection
from tkinter import *  # GUI framework
from tkinter import ttk, messagebox  # Advanced widgets and dialogs
from PIL import Image, ImageTk  # Image processing for GUI
import pygame  # Audio playback (mixer)
from pygame import mixer  # Audio mixing initialization
from concurrent.futures import ThreadPoolExecutor  # Parallel processing
import threading  # Thread synchronization for TTS
import winsound  # Windows beep sound for recording prompt
import sys  # System operations and PyInstaller support

# Handle PyInstaller bundled environment

if getattr(sys, 'frozen', False): # When you convert this Python script to a standalone .exe file using PyInstaller
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize pygame mixer
mixer.init()

# =================== Initialization ===================
engine = pyttsx3.init()
voices = engine.getProperty('voices')
try:
    engine.setProperty('voice', voices[1].id) # Use female voice (index 1)
except IndexError:
    print("Voice index 1 not found. Using default voice.")
engine.setProperty('rate', 150)     # Speaking speed (words per minute)

r = sr.Recognizer()
r.energy_threshold = 4000  # Microphone sensitivity
r.dynamic_energy_threshold = False  # Fixed threshold for consistent behavior
tts_lock = threading.Lock()  # Lock for thread-safe TTS access

# Preload sentence transformer model with PyInstaller support
try:
    model_sentence = SentenceTransformer('all-MiniLM-L6-v2') 
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to local copy if available
    local_model_path = os.path.join(bundle_dir, 'all-MiniLM-L6-v2')
    if os.path.exists(local_model_path):
        model_sentence = SentenceTransformer(local_model_path)
    else:
        raise RuntimeError("Could not load sentence transformer model")

GEMINI_API_KEY = "AIzaSyBfS_YYa262erJX_q-kwPzRZIJ5tcKGS2w"
genai.configure(api_key=GEMINI_API_KEY)

# Define emotions list
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Create a simple emotion model as fallback
def create_simple_emotion_model():
    """Create a simple emotion detection model"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Load models
def load_models():
    try:
        model_path = os.path.join(bundle_dir, 'model_file_30epochs.h5') if getattr(sys, 'frozen', False) else r"D:\virtual_interview_simulator\AAA\ALL\emotion_model.h5"
        
        # If model file doesn't exist, create a simple model
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}. Creating a simple model.")
            emotion_model = create_simple_emotion_model()
            # Save the simple model for future use
            emotion_model.save(model_path)
            print("Created and saved a simple emotion model")
        else:
            # Try to load the existing model
            try:
                emotion_model = tf.keras.models.load_model(model_path)
                print("Successfully loaded existing emotion model")
            except Exception as e:
                print(f"Error loading existing model: {e}. Creating a new simple model.")
                emotion_model = create_simple_emotion_model()
                emotion_model.save(model_path)
                print("Created and saved a new simple emotion model")
        
        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            # Try alternative path
            cascade_path = os.path.join(bundle_dir, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(cascade_path):
                messagebox.showerror("Error", f"Haar cascade file not found at: {cascade_path}")
                return None, None
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            messagebox.showerror("Error", "Failed to load Haar cascade classifier")
            return None, None
            
        return emotion_model, face_cascade
        
    except Exception as e:
        print(f"Error loading models: {e}")
        messagebox.showerror("Error", f"Could not load emotion detection models: {e}")
        return None, None

# Load models
emotion_model, face_cascade = load_models()

class DataCache:
    def __init__(self):
        self.sample_answers = {}
        self._load_sample_answers()
        
    def _load_sample_answers(self):
        try:
            sample_file = os.path.join(bundle_dir, 'sample_answers.json') if getattr(sys, 'frozen', False) else os.path.join(os.path.dirname(__file__), "sample_answers.json")
            if not os.path.exists(sample_file):
                messagebox.showerror("Error", "sample_answers.json not found!")
                return False
            
            with open(sample_file, 'r', encoding='utf-8') as f:
                self.sample_answers = json.load(f)
                
            if "General HR Round" not in self.sample_answers:
                self.sample_answers["General HR Round"] = {}
                
            for question in BASIC_HR_QUESTIONS:
                if question not in self.sample_answers["General HR Round"]:
                    self.sample_answers["General HR Round"][question] = "This is a sample answer for " + question
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load sample answers: {e}")
            return False

BASIC_HR_QUESTIONS = [
    "Tell me about yourself",
    "What are your strengths?",
    "Why do you want to join our company?",
    "Where do you see yourself in five years?",
    "What is your greatest professional achievement?"
]

class InterviewSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Interview Simulator")
        self.root.state('zoomed')
        
        # Define color scheme
        self.colors = {
            'primary': "#2709ea",
            'secondary': "#2B1169",
            'accent': "#7878f4",
            'background': "#5c55e9",
            'text': "#48B9A8",
            'success': "#7be27e",
            'warning': "#ffb23e",
            'danger': "#f75348",
            'light': "#ffffff",
            'dark': "#5C5C5C",
            'highlight': "#fff27a"
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.data_cache = DataCache()
        if not self.data_cache.sample_answers:
            self.root.destroy()
            return
            
        # Load background image
        self.background_label = None  # Initialize before use
        try:
            bg_path = os.path.join(bundle_dir, 'anirban.png') if getattr(sys, 'frozen', False) else r"D:\virtual_interview_simulator\AAA\ALL\anirban.png"
            self.bg_image = Image.open(bg_path)
            self.bg_image = self.bg_image.resize((self.root.winfo_screenwidth(), 
                                               self.root.winfo_screenheight()), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.background_label = Label(self.root, image=self.bg_photo)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Could not load background image: {e}")
            self.root.configure(bg=self.colors['background'])
        
        # Variables
        self.selected_domain = StringVar()
        self.current_question = StringVar(value="Question will appear here...")
        self.score = StringVar(value="0%")
        self.emotion = StringVar(value="Not detected")
        self.rule_feedback = StringVar(value="Feedback will appear here...")
        self.gemini_feedback = StringVar(value="Gemini feedback will appear here...")
        self.interview_active = False
        self.webcam_active = False
        self.cap = None
        self.current_emotion = "neutral"
        self.last_emotion_time = 0
        self.emotion_cache = "neutral"
        self.webcam_frame_cache = None
        self.last_frame_time = 0
        self.answer_processed = BooleanVar(value=True)
        self.current_question_index = 0
        self.questions = []
        self.user_answer = ""
        self.emotion_model = emotion_model
        self.face_cascade = face_cascade
        
        self.create_welcome_frame()
        
    def create_welcome_frame(self):
        for widget in self.root.winfo_children():
            if self.background_label is not None and widget != self.background_label:
                widget.destroy()
            elif self.background_label is None:
                widget.destroy()
        
        welcome_frame = Frame(self.root, bg=self.colors['light'], bd=2, relief=GROOVE)
        welcome_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
        
        Label(welcome_frame, text="Welcome to Virtual Interview Simulator", 
             font=("Helvetica", 24, "bold"), bg=self.colors['light'], 
             fg=self.colors['secondary']).pack(pady=20)
        
        # Speak in background thread to avoid blocking GUI rendering
        self.root.after(500, lambda: self.executor.submit(self.speak, "Welcome to the virtual interview simulator. Please select a domain to begin."))
        
        domain_frame = Frame(welcome_frame, bg=self.colors['light'])
        domain_frame.pack(pady=20)
        
        Label(domain_frame, text="Select Interview Domain:", 
              font=("Helvetica", 14), bg=self.colors['light'], 
              fg=self.colors['text']).pack()
        
        domains = list(self.data_cache.sample_answers.keys())
        domain_menu = ttk.Combobox(domain_frame, textvariable=self.selected_domain, 
                                  values=domains, font=("Helvetica", 12), width=40)
        domain_menu.pack(pady=10)
        domain_menu.current(0)
        
        start_btn = Button(welcome_frame, text="Start Interview", command=self.start_interview_setup,
                          font=("Helvetica", 14), bg=self.colors['primary'], fg=self.colors['light'], 
                          activebackground=self.colors['secondary'], padx=20, pady=10)
        start_btn.pack(pady=30)
        
    def start_interview_setup(self):
        if not self.selected_domain.get():
            messagebox.showwarning("Warning", "Please select a domain first!")
            return
            
        for widget in self.root.winfo_children():
            if self.background_label is not None and widget != self.background_label:
                widget.destroy()
            elif self.background_label is None:
                widget.destroy()
        
        self.create_interview_interface()
        
    def create_interview_interface(self):
        main_frame = Frame(self.root, bg=self.colors['light'], bd=2, relief=GROOVE)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        top_frame = Frame(main_frame, bg=self.colors['light'])
        top_frame.pack(fill=BOTH, expand=True)
        
        webcam_frame = Frame(top_frame, bg=self.colors['dark'], bd=2, relief=GROOVE)
        webcam_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
        
        self.webcam_label = Label(webcam_frame, text="Webcam Feed - Click 'Start Webcam'", 
                                bg=self.colors['dark'], fg=self.colors['light'],
                                font=("Helvetica", 12))
        self.webcam_label.pack(fill=BOTH, expand=True)
        
        control_frame = Frame(top_frame, bg=self.colors['light'], bd=2, relief=GROOVE)
        control_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)
        
        Label(control_frame, text="Interview Controls", font=("Helvetica", 14, "bold"), 
              bg=self.colors['light'], fg=self.colors['secondary']).pack(pady=10)
        
        self.start_webcam_btn = Button(control_frame, text="Start Webcam", command=self.start_webcam,
                                     font=("Helvetica", 12), bg=self.colors['success'], 
                                     fg=self.colors['light'], 
                                     activebackground=self.colors['secondary'], 
                                     padx=15, pady=8)
        self.start_webcam_btn.pack(pady=10, fill=X)
        
        self.start_interview_btn = Button(control_frame, text="Start Interview", 
                                        command=self.start_interview_process,
                                        font=("Helvetica", 12), bg=self.colors['primary'], 
                                        fg=self.colors['light'], 
                                        activebackground=self.colors['secondary'], 
                                        padx=15, pady=8, state=DISABLED)
        self.start_interview_btn.pack(pady=10, fill=X)
        
        self.stop_btn = Button(control_frame, text="Stop Interview", command=self.stop_interview,
                              font=("Helvetica", 12), bg=self.colors['danger'], 
                              fg=self.colors['light'], 
                              activebackground=self.colors['secondary'],
                              padx=15, pady=8, state=DISABLED)
        self.stop_btn.pack(pady=10, fill=X)
        
        question_frame = LabelFrame(control_frame, text="Current Question", 
                                  font=("Helvetica", 12, "bold"), 
                                  bg=self.colors['light'], fg=self.colors['text'])
        question_frame.pack(pady=20, fill=X)
        
        self.question_text = Text(question_frame, height=6, width=40, wrap=WORD, 
                                font=("Helvetica", 11))
        self.question_text.pack(fill=BOTH, expand=True)
        self.question_text.insert(END, "Question will appear here...")
        self.question_text.config(state=DISABLED)
        
        answer_frame = LabelFrame(control_frame, text="My Answer", 
                                font=("Helvetica", 12, "bold"), 
                                bg=self.colors['light'], fg=self.colors['text'])
        answer_frame.pack(pady=(0, 20), fill=X)
        
        self.answer_text = Text(answer_frame, height=6, width=40, wrap=WORD, 
                              font=("Helvetica", 11))
        self.answer_text.pack(fill=BOTH, expand=True)
        self.answer_text.insert(END, "Your answer will appear here...")
        self.answer_text.config(state=DISABLED)
        
        bottom_frame = Frame(main_frame, bg=self.colors['light'], bd=2, relief=GROOVE)
        bottom_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
        
        result_frame = LabelFrame(bottom_frame, text="Interview Results", 
                                font=("Helvetica", 14, "bold"), 
                                bg=self.colors['light'], fg=self.colors['secondary'])
        result_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)
        
        result_grid = Frame(result_frame, bg=self.colors['light'])
        result_grid.pack(pady=10, padx=10, fill=BOTH)
        
        Label(result_grid, text="Your Score:", font=("Helvetica", 12, "bold"), 
             bg=self.colors['light'], fg=self.colors['text']).grid(row=0, column=0, sticky=W, pady=5)
        self.score_label = Label(result_grid, textvariable=self.score, font=("Helvetica", 12), 
             bg=self.colors['light'], fg=self.colors['primary'])
        self.score_label.grid(row=0, column=1, sticky=W, pady=5)
        
        Label(result_grid, text="Detected Emotion:", font=("Helvetica", 12, "bold"), 
             bg=self.colors['light'], fg=self.colors['text']).grid(row=1, column=0, sticky=W, pady=5)
        Label(result_grid, textvariable=self.emotion, font=("Helvetica", 12), 
             bg=self.colors['light'], fg=self.colors['text']).grid(row=1, column=1, sticky=W, pady=5)
        
        feedback_frame = LabelFrame(bottom_frame, text="Feedback", 
                                  font=("Helvetica", 14, "bold"), 
                                  bg=self.colors['light'], fg=self.colors['secondary'])
        feedback_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)
        
        rule_frame = Frame(feedback_frame, bg=self.colors['light'])
        rule_frame.pack(fill=X, pady=5)
        Label(rule_frame, text="Rule-Based Feedback:", font=("Helvetica", 12, "bold"), 
             bg=self.colors['light'], fg=self.colors['text']).pack(side=LEFT, anchor=W)
        
        self.rule_feedback_label = Label(feedback_frame, textvariable=self.rule_feedback, 
                            font=("Helvetica", 11), bg=self.colors['light'], 
                            fg=self.colors['text'], wraplength=400, justify=LEFT)
        self.rule_feedback_label.pack(fill=X, padx=10, pady=5)
        
        gemini_frame = Frame(feedback_frame, bg=self.colors['light'])
        gemini_frame.pack(fill=X, pady=5)
        Label(gemini_frame, text="Gemini Feedback:", font=("Helvetica", 12, "bold"), 
             bg=self.colors['light'], fg=self.colors['text']).pack(side=LEFT, anchor=W)
        
        self.gemini_feedback_label = Label(feedback_frame, textvariable=self.gemini_feedback, 
                              font=("Helvetica", 11), bg=self.colors['light'], 
                              fg=self.colors['text'], wraplength=400, justify=LEFT)
        self.gemini_feedback_label.pack(fill=X, padx=10, pady=5)
        
    def start_webcam(self):
        if not self.webcam_active:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam!")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 24)
            
            self.webcam_active = True
            self.start_webcam_btn.config(text="Webcam Active", state=DISABLED)
            self.start_interview_btn.config(state=NORMAL)
            self.update_webcam()
            
    def update_webcam(self):
        if self.webcam_active and self.cap.isOpened():
            start_time = time.time()
            
            if (start_time - self.last_frame_time) < 0.03:
                self.root.after(10, self.update_webcam)
                return
                
            ret, frame = self.cap.read()
            if ret:
                self.executor.submit(self.process_frame, frame.copy(), start_time)
                
            self.last_frame_time = start_time
            self.root.after(10, self.update_webcam)
        elif self.webcam_active:
            self.webcam_active = False
            self.start_webcam_btn.config(text="Start Webcam", state=NORMAL)
            
    def process_frame(self, frame, frame_time):
        if self.interview_active and (frame_time - self.last_emotion_time) > 0.5:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                                    minNeighbors=5, 
                                                    minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                padding = int(w * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = gray[y:y + h, x:x + w]
                
                # Only try emotion detection if model is loaded
                if self.emotion_model is not None:
                    try:
                        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
                        normalized_face = resized_face / 255.0
                        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
                        prediction = self.emotion_model.predict(reshaped_face, verbose=0)
                        emotion_index = np.argmax(prediction)
                        self.emotion_cache = emotions[emotion_index]
                        
                        self.root.after(0, lambda: self.emotion.set(self.emotion_cache.capitalize()))
                        
                        cv2.putText(frame, self.emotion_cache.capitalize(), (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    except Exception as e:
                        print(f"Emotion detection error: {e}")
                        cv2.putText(frame, "Model Error", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No Model", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            self.last_emotion_time = frame_time
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        max_width = 640
        max_height = 480
        
        img_ratio = img.width / img.height
        
        if img_ratio > 1:
            new_width = min(img.width, max_width)
            new_height = int(new_width / img_ratio)
        else:
            new_height = min(img.height, max_height)
            new_width = int(new_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.root.after(0, lambda: self.update_webcam_display(imgtk, new_width, new_height))
        
    def update_webcam_display(self, imgtk, width, height):
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)
        self.webcam_label.config(width=width, height=height)
            
    def start_interview_process(self):
        if not self.webcam_active:
            messagebox.showwarning("Warning", "Please start the webcam first!")
            return
            
        self.interview_active = True
        self.start_interview_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        
        # Prepare questions list
        self.questions = BASIC_HR_QUESTIONS[:3]
        domain_questions = list(self.data_cache.sample_answers[self.selected_domain.get()].keys())
        self.questions.extend(domain_questions)
        
        self.current_question_index = 0
        
        # Clear all sections
        self.root.after(0, lambda: [
            self.question_text.config(state=NORMAL),
            self.question_text.delete(1.0, END),
            self.question_text.insert(END, "Let's start the interview!"),
            self.question_text.config(state=DISABLED),
            self.answer_text.config(state=NORMAL),
            self.answer_text.delete(1.0, END),
            self.answer_text.insert(END, "Your answers will appear here..."),
            self.answer_text.config(state=DISABLED),
            self.score.set("0%"),
            self.rule_feedback.set("Feedback will appear here..."),
            self.gemini_feedback.set("Gemini feedback will appear here...")
        ])
        
        self.executor.submit(self.speak, "Let's start the interview!")
        
        self.root.after(2000, self.ask_next_question)
        
    def ask_next_question(self):
        if not self.interview_active:
            return
            
        if self.current_question_index >= len(self.questions):
            def _finish_interview():
                self.speak("The interview has concluded. Thank you for participating!")
                messagebox.showinfo("Interview Complete", "The interview has concluded. Thank you!")
                self.stop_interview()
            self.root.after(0, _finish_interview)
            return
        
        question = self.questions[self.current_question_index]
        
        self.root.after(0, lambda: [
            self.question_text.config(state=NORMAL),
            self.question_text.delete(1.0, END),
            self.question_text.insert(END, question),
            self.question_text.config(state=DISABLED),
            self.answer_text.config(state=NORMAL),
            self.answer_text.delete(1.0, END),
            self.answer_text.insert(END, "Please answer after the beep..."),
            self.answer_text.config(state=DISABLED),
            self.score.set("0%"),
            self.rule_feedback.set("Feedback will appear here..."),
            self.gemini_feedback.set("Gemini feedback will appear here...")
        ])
        
        self.root.after(1000, lambda: self.speak_question(question))
    
    def speak_question(self, question):
        def _speak_then_prompt():
            self.speak(question)
            self.root.after(1000, lambda: self.prompt_for_answer(question))
        self.executor.submit(_speak_then_prompt)
    
    def prompt_for_answer(self, question):
        def _prompt_then_record():
            self.speak("Please answer now")
            try:
                winsound.Beep(1000, 500)
            except:
                pass
            self.root.after(0, lambda: self.record_answer(question))
        self.executor.submit(_prompt_then_record)
    
    def record_answer(self, question):
        self.root.after(0, lambda: [
            self.answer_text.config(state=NORMAL),
            self.answer_text.delete(1.0, END),
            self.answer_text.insert(END, "Recording your answer..."),
            self.answer_text.config(state=DISABLED)
        ])

        # Run audio recording and processing in background thread
        self.executor.submit(self._record_and_process, question)

    def _record_and_process(self, question):
        audio_file = self.record_audio(duration=10)
        if audio_file:
            self.process_question_answer(question, audio_file)


    def process_question_answer(self, question, audio_file):
        self.user_answer = self.recognize_speech(audio_file)
        
        self.root.after(0, lambda: [
            self.answer_text.config(state=NORMAL),
            self.answer_text.delete(1.0, END),
            self.answer_text.insert(END, self.user_answer),
            self.answer_text.config(state=DISABLED)
        ])
        
        domain = "General HR Round" if self.current_question_index < len(BASIC_HR_QUESTIONS[:3]) else self.selected_domain.get()
        correct_answer = self.data_cache.sample_answers[domain][question]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            score_future = executor.submit(self.evaluate_answer, self.user_answer, correct_answer)
            rule_fb_future = executor.submit(self.rule_based_feedback, self.user_answer)
            gemini_fb_future = executor.submit(self.get_gemini_feedback, self.user_answer, question, self.emotion_cache)
            
            score = score_future.result()
            rule_feedback = rule_fb_future.result()
            gemini_fb = gemini_fb_future.result()
        
        self.root.after(0, lambda: [
            self.score.set(f"{score}%"),
            self.score_label.config(fg="green" if score > 70 else "orange" if score > 40 else "red"),
            self.rule_feedback.set(rule_feedback),
            self.gemini_feedback.set(gemini_fb)
        ])
        
        self.log_to_csv(domain, question, correct_answer, self.user_answer, score, 
                      rule_feedback, gemini_fb, self.emotion_cache)
        
        try:
            os.remove(audio_file)
        except:
            pass
            
        self.current_question_index += 1
        self.root.after(3000, self.ask_next_question)
            
    def stop_interview(self):
        self.interview_active = False
        self.root.after(0, lambda: [
            self.stop_btn.config(state=DISABLED),
            self.start_interview_btn.config(state=NORMAL)
        ])
        
        if self.webcam_active and self.cap:
            self.cap.release()
            self.webcam_active = False
            self.root.after(0, lambda: self.start_webcam_btn.config(text="Start Webcam", state=NORMAL))
        
    def record_audio(self, duration=10):
        fs = 44100
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        filename = temp_file.name
        temp_file.close()

        try:
            # Use mono (1 channel) – avoids PortAudio error
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            write(filename, fs, recording)
        except Exception as e:
            print(f"Audio recording failed: {e}")
            return None

        return filename


        
    def recognize_speech(self, filename='output.wav'):
        if not os.path.exists(filename):
            return f"Error: Audio file not found at {filename}"
            
        try:
            with sr.AudioFile(filename) as source:
                audio = r.record(source)
                try:
                    return r.recognize_google(audio)
                except sr.UnknownValueError:
                    return "Sorry, I could not understand the audio. Please speak clearly."
                except sr.RequestError as e:
                    return f"Speech recognition error: {e}"
        except Exception as e:
            return f"Error processing audio file: {e}"
            
    def evaluate_answer(self, user_answer, correct_answer):
        user_embedding = model_sentence.encode(user_answer, convert_to_tensor=True)
        correct_embedding = model_sentence.encode(correct_answer, convert_to_tensor=True)
        score = util.pytorch_cos_sim(user_embedding, correct_embedding).item()
        return round(score * 100, 2)
        
    def rule_based_feedback(self, user_answer):
        blob = TextBlob(user_answer)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        grammar_issues = len(blob.correct().words) - len(user_answer.split())
        feedback = []
        if grammar_issues > 0:
            feedback.append("Some grammatical errors were detected.")
        if polarity < -0.2:
            feedback.append("The tone seemed negative. Try to stay positive.")
        elif polarity > 0.5:
            feedback.append("Good enthusiastic tone!")
        if subjectivity > 0.6:
            feedback.append("Answer seems too opinion-based. Try to be more factual.")
        elif subjectivity < 0.4:
            feedback.append("Well-balanced factual response.")
        return "; ".join(feedback) or "Looks good overall."
        
    def get_gemini_feedback(self, user_answer, question, emotion):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"You are an expert interview evaluator.\n"
                f"Question: {question}\n"
                f"Candidate's Answer: {user_answer}\n"
                f"Detected Emotion: {emotion}\n"
                f"Give short and concise feedback (max 2-3 sentences) on:\n"
                f"- Content accuracy\n"
                f"- Clarity\n"
                f"- Professional tone\n"
                f"- Emotional appropriateness for interview"
            )
            response = model.generate_content(prompt)
            return response.text.strip() if hasattr(response, 'text') else "Gemini response not available."
        except Exception as e:
            return f"Error from Gemini API: {e}"
            
    def log_to_csv(self, domain, question, correct_answer, user_answer, score, rule_feedback, gemini_fb, emotion):
        log_file = os.path.join(bundle_dir, 'interview_log.csv') if getattr(sys, 'frozen', False) else "interview_log.csv"
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                domain, question, correct_answer,
                user_answer, score, rule_feedback,
                gemini_fb, emotion
            ])
            
    def speak(self, text):
        try:
            with tts_lock:
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the interview simulator?"):
            self.stop_interview()
            self.executor.shutdown(wait=False)
            self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = InterviewSimulator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()