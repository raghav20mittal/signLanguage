
# For capturing hand coordinates
import cv2
import mediapipe as mp

# For processing data
import pandas as pd
import numpy as np
dataset = pd.read_csv('C:\Users\Lenovo\Desktop\handsign_project\hand_dataset_1000_24.csv')

# Show dataset first five data
dataset.head()


# Defining X and Y from dataset for training and testing
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values
from sklearn.model_selection import train_test_split


# We will take 33% from 1000 for our test data.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
# Standardize dataset


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print('The accuracy of model is:')
print(accuracy_score(y_test, y_pred)*100)


error = []


# Calculating error for K values between 1 and 40
for i in range(1, 40):
   knn = KNeighborsClassifier(n_neighbors=i)
   knn.fit(X_train, y_train)
   pred_i = knn.predict(X_test)
   from sklearn.metrics import mean_absolute_error as mae
   error.append(np.mean(pred_i != y_test))

# import matplotlib.pyplot as plt


# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
#         markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Error')
# plt.show()


# Initialize mediapipe hand


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
print("------here-----")
# Initialize mediapipe hand capture webcam


cap = cv2.VideoCapture(0)
print("------here1-----")

with mp_hands.Hands(
   max_num_hands = 1,
   min_detection_confidence=0.5,
   min_tracking_confidence=0.5) as hands:
   while cap.isOpened():
       success, image = cap.read()


       if not success:
           print("Ignoring empty camera frame.")
           # If loading a video, use 'break' instead of 'continue'.
           continue


       # Flip the image horizontally for a later selfie-view display, and convert
       # the BGR image to RGB.
       image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)


       # To improve performance, optionally mark the image as not writeable to
       # pass by reference.
       image.flags.writeable = False
       results = hands.process(image)


       # Draw the hand annotations on the image.
       image.flags.writeable = True
       image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       if results.multi_hand_landmarks:
           for hand_landmarks in results.multi_hand_landmarks:
               coords = hand_landmarks.landmark
               mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
               coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
               coords = scaler.transform([coords])
              
               # Alternative for dataset using z coordinates.
               # Z coordinates is not recommended, since you need to adjust your distance from camera.
#                 coords = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in coords]).flatten())
              
               predicted = classifier.predict(coords)


           # Get status box
           cv2.rectangle(image, (0,0), (100, 60), (245, 90, 16), -1)


           # Display Class
           cv2.putText(image, 'CLASS'
                       , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
           cv2.putText(image, str(predicted[0])
                       , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


       cv2.imshow('MediaPipe Hands', image)


       # Press esc to close webcam
       if cv2.waitKey(5) & 0xFF == 27:
           break
cap.release()
cv2.destroyAllWindows()