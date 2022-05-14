from System_Status import System_Status
from keras.models import load_model
from PIL import Image, ImageTk
import Jetson.GPIO as GPIO1
import RPi.GPIO as GPIO2
import tkinter as tk
import numpy as np
import threading
import warnings
import cv2
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = INFO, WARNING, and ERROR
warnings.filterwarnings('ignore')

########################################################################################################################
# List of Module functions: -
########################################################################################################################
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(img):
    return cv2.equalizeHist(img)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def get_class_name(class_number):
    if class_number == 0:
        return " Safe Driving"
    elif class_number == 1:
        return " [Distracted] Texting - Right"
    elif class_number == 2:
        return " [Distracted] Talking On The Phone - Right"
    elif class_number == 3:
        return " [Distracted] Texting - Left"
    elif class_number == 4:
        return " [Distracted] Talking On The Phone - Left"
    elif class_number == 5:
        return " [Distracted] Operating The Radio"
    elif class_number == 6:
        return " [Distracted] Drinking"
    elif class_number == 7:
        return " [Distracted] Reaching Behind"
    elif class_number == 8:
        return " [Distracted] Hair And Makeup"
    elif class_number == 9:
        return " [Distracted] Talking To Passenger"


def load_selected_model():
    # IMPORT THE TRAINED MODEL
    model_path = 'saved_models/selected_model/Model_1'
    return load_model(model_path)


def set_warning_status(warning_status):
    if warning_status != System_Status.warning_status:
        System_Status.warning_status = warning_status
        GPIO2.output(GPIO_pin_number, warning_status)


def detect_distraction():
    model = load_selected_model()

    counter = 1
    while True:
        # 1. READ IMAGE FROM CAMERA
        success, original_image = capture.read()
        if success:
            # 2. PROCESS IMAGE
            img = np.asarray(original_image)
            img = cv2.resize(img, (64, 64))
            img = preprocessing(img)
            img = img.reshape(-1, 64, 64, 1)
            cv2.putText(original_image, "Driver Status: ", (20, 35), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if System_Status.status != "Pause":
                # 3. PREDICT IMAGE
                predictions = model.predict(img, batch_size=40)

                class_index = np.argmax(predictions)
                probability_value = np.amax(predictions)

                if probability_value >= threshold:
                    System_Status.current_driver_status = get_class_name(class_index)
                    if System_Status.current_driver_status != System_Status.driver_status:
                        System_Status.driver_status = System_Status.current_driver_status
                        try:
                            output_video.write(original_image)
                        except:
                            pass

                    if class_index != 0:
                        set_warning_status(True)
                    else:
                        set_warning_status(False)

                    cv2.putText(original_image, "    " + System_Status.driver_status, (120, 35), font, 1,
                                (0, 0, 255) if class_index != 0 else (0, 235, 0), 2, cv2.LINE_AA)
            else:
                period = ""

                if counter > 10:
                    counter = 1
                for i in range(counter):
                    period += "."
                counter = counter + 1
                cv2.putText(original_image, "    " + "Detection is paused " + period, (120, 35), font, 1,
                            (46, 15, 207), 2, cv2.LINE_AA)

            # Inserting the current frame into the GUI default right-hand side frame
            frame = cv2.resize(original_image, (frame_width, frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            default_image.paste(Image.fromarray(frame))
        else:
            print("Cannot connect to the camera")
            exit(-1)


def start_detecting():
    start_button.config(state='disabled')
    shutdown_button.config(state='normal')
    pause_button.config(state='normal')
    System_Status.status = "Start"
    try:
        threading.Thread(target=detect_distraction, daemon=True).start()
    except threading.ThreadError:
        pass


def shutdown_detecting():
    GPIO2.cleanup()
    System_Status.status = "Shutdown"
    exit(1)


def pause_detecting():
    pause_button.config(state='disabled')
    resume_button.config(state='normal')
    System_Status.status = "Pause"
    try:
        threading.Thread(target=detect_distraction, daemon=True).start()
    except threading.ThreadError:
        pass


def resume_detecting():
    pause_button.config(state='normal')
    resume_button.config(state='disabled')
    System_Status.status = "Resume"
    System_Status.driver_status += " "
    try:
        threading.Thread(target=detect_distraction, daemon=True).start()
    except threading.ThreadError:
        pass


########################################################################################################################

########################################################################################################################
# The start point of the system will be executed after this line (Driver Code): -
########################################################################################################################

################################################################################
# General variables initialization
camera_number = 0
frame_width = 640  # CAMERA RESOLUTION
frame_height = 480
video_size = (frame_width, frame_height)
brightness = 180
threshold = 0.7  # THE MINIMUM PROBABILITY THRESHOLD (ACCEPTANCE PERCENTAGE)
font = cv2.FONT_HERSHEY_PLAIN
frames_per_second = 30
#################################################################################

#################################################################################
# SETUP THE VIDEO CAMERA
capture = cv2.VideoCapture(camera_number)
capture.set(3, frame_width)
capture.set(4, frame_height)
capture.set(10, brightness)
#################################################################################

#################################################################################
# SAVE THE DETECTION PART OF THE RESULT VIDEO: -
output_video = cv2.VideoWriter(
    'Testing-result.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    frames_per_second,
    video_size
)
#################################################################################

#################################################################################
# SETUP THE AURAL ALARM: -
GPIO_pin_number = 12
GPIO2.setmode(GPIO2.BOARD)
GPIO2.setup(GPIO_pin_number, GPIO2.OUT)
#################################################################################

#################################################################################
# GUI Implementation is starting from the following lines: -
#################################################################################

#################################################################################
# GUI VARIABLES: -
title = 'User\'s Actions'
UI_font = "Cambria 24 bold"
background_color = "#A9F8BA"
start_image_path = "images/start.png"
shutdown_image_path = "images/shutdown.png"
resume_image_path = "images/resume.png"
pause_image_path = "images/pause.png"
#################################################################################

root = tk.Tk()  # Creating the main window (Beginning of the UI creation).

root.overrideredirect(True)  # Remove the window title bar.
root.config(bg=background_color)

screen_width = (root.winfo_screenwidth() / 2) - ((frame_width * 2) / 2)  # TO CENTER THE WINDOW HORIZONTALLY
screen_height = (root.winfo_screenheight() / 2) - (frame_height / 2)  # TO CENTER THE WINDOW VERTICALLY

# SETUP OF THE WINDOW WIDTH, HEIGHT, and POSITION OF THE x & y POINTS (@screen_width, @screen_height)
root.geometry("{}x{}+{}+{}".format(frame_width * 2, frame_height, int(screen_width), int(screen_height)))

buttons_frame = tk.Frame(root, bg=background_color, padx=56)
buttons_frame.pack(side=tk.LEFT)

title_label = tk.Label(buttons_frame, font=UI_font, text=title, bg=background_color)
title_label.grid(row=0, column=4, padx=10, pady=10)

start_image = tk.PhotoImage(file=start_image_path)
start_button = tk.Button(buttons_frame, bg=background_color, image=start_image, borderwidth=0, command=start_detecting)
start_button.grid(row=2, column=1, columnspan=3, padx=10, pady=10)

shutdown_image = tk.PhotoImage(file=shutdown_image_path)
shutdown_button = tk.Button(buttons_frame, bg=background_color,
                            image=shutdown_image, borderwidth=0,
                            command=shutdown_detecting)

shutdown_button.grid(row=2, column=8, padx=10, pady=10)

resume_image = tk.PhotoImage(file=resume_image_path)
resume_button = tk.Button(buttons_frame, bg=background_color,
                          image=resume_image, borderwidth=0,
                          state='disabled', command=resume_detecting)

resume_button.grid(row=4, column=1, columnspan=3, padx=10, pady=10)

pause_image = tk.PhotoImage(file=pause_image_path)
pause_button = tk.Button(buttons_frame, bg=background_color,
                         image=pause_image, borderwidth=0,
                         state='disabled', command=pause_detecting)

pause_button.grid(row=4, column=8, padx=10, pady=10)

# SET A DEFAULT BACKGROUND FOR THE RESULT VIDEO TILL THE USER CLICK THE START BUTTON.
default_image = ImageTk.PhotoImage(Image.new('RGB', video_size, (50, 30, 80)))
vbox = tk.Label(root, image=default_image, width=frame_width, height=frame_height)
vbox.pack(side=tk.RIGHT)

root.mainloop()  # Ending of the UI Creation.
