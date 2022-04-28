import numpy as np
import cv2
import pickle
import tensorflow
from keras.models import load_model

################################################################################
# General variables initialization
from matplotlib import pyplot as plt

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.90  # THE MINIMUM PROBABILITY THRESHOLD (ACCEPTANCE PERCENTAGE)
font = cv2.FONT_HERSHEY_PLAIN
#################################################################################

#################################################################################
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
#################################################################################

#################################################################################
# IMPORT THE TRAINED MODEL
# pickle_in = open("saved_model.p", "rb")  # READ BYTE
model = tensorflow.keras.models.load_model('saved_models/Final model')


# model.load_weights('saved_models/Last model adam/weights')


#################################################################################
activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(img):
    return cv2.equalizeHist(img)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255

    return img


def getClassName(classNo):
    if   classNo == 0:
        return "Safe Driving"
    elif classNo == 1:
        return "Texting - Right"
    elif classNo == 2:
        return "Talking On The Phone - Right"
    elif classNo == 3:
        return "Texting - Left"
    elif classNo == 4:
        return "Talking On The Phone - Left"
    elif classNo == 5:
        return "Operating The Radio"
    elif classNo == 6:
        return "Drinking"
    elif classNo == 7:
        return "Reaching Behind"
    elif classNo == 8:
        return "Hair And Makeup"
    elif classNo == 9:
        return "Talking To Passenger"

count = 0
while True:
    if count > 5:
        break
    # 1. READ IMAGE
    success, imgOriginal = cap.read()

    # 2. PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv2.flip(img, -1)
    img = cv2.resize(img, (64, 64))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    plt.imshow(img, cmap='gray')
    img = img.reshape(-1, 64, 64, 1)
    cv2.putText(imgOriginal, "CLASS:", (20, 35), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY:", (20, 75), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    plt.show()
    # 3. PREDICT IMAGE
    predictions = model.predict(img, batch_size=40, verbose=1)
    # predictions = np.argmax(model.predict(img), axis=-1)
    # print(np.size(predictions))
    # print(predictions)

    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)
    print(getClassName(classIndex))

    if probabilityValue >= threshold:
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1



def prepare(filepath, model):

    img_brute = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_brute = cv2.resize(img_brute,(64,64))
    plt.imshow(img_brute, cmap='gray')

    new_img = img_brute.reshape(-1,64,64,1)

    y_prediction = model.predict(new_img, batch_size=40, verbose=1)
    print('Y prediction: {}'.format(y_prediction))
    print('Predicted: {}'.format(activity_map.get('c{}'.format(np.argmax(y_prediction)))))

    plt.show()

prepare('./imgTesting/own/photo_2022-04-28_03-15-31 (2).jpg', model)
# img_brute = cv2.imread('./imgTesting/img_93603.jpg', cv2.IMREAD_GRAYSCALE)
# img_brute = cv2.resize(img_brute,(64,64))
# new_img = img_brute.reshape(-1,64,64,1)
# test = model.predict(new_img)
# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()