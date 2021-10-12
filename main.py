import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'people_faces'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)

for cl in my_list:
    current_image = cv2.imread(f'{path}/{cl}')
    images.append(current_image)
    class_names.append(os.path.splitext(cl)[0])

print(class_names)


def find_encodings(images_fn):
    encoded_list = []
    for image_fn in images_fn:
        image_fn = cv2.cvtColor(image_fn, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image_fn)[0]
        encoded_list.append(encode)
    return encoded_list


def mark_attendance(name):
    with open('attendance_list.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])

        if name not in name_list:
            now = datetime.now()
            date_time_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_time_string}')


encoded_list_known = find_encodings(images)
print('Encoding of Images Done')
print("Now Starting Webcam")

cap = cv2.VideoCapture(1)

while True:
    success, image = cap.read()
    small_image = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(small_image)
    encoding_of_current_frame = face_recognition.face_encodings(small_image, faces_current_frame)

    for encoded_face, face_location in zip(encoding_of_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encoded_list_known, encoded_face)
        facial_distance = face_recognition.face_distance(encoded_list_known, encoded_face)
        # print(facial_distance)

        match_index = np.argmin(facial_distance)

        if matches[match_index]:
            name = class_names[match_index].upper()
            print(name)

            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            mark_attendance(name)

    cv2.imshow('Webcam', image)
    cv2.waitKey(1)
