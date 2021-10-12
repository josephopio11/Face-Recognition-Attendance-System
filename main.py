import cv2
import numpy as np
import face_recognition
import os

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


encoded_list_known = find_encodings(images)
print('Encoding of Images Done')

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
        print(facial_distance)

        match_index = np.argmin(facial_distance)

        if matches[match_index]:
            name = class_names[match_index].upper()
            print(name)
