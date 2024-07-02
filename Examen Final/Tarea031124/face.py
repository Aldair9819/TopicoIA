import face_recognition
from PIL import ImageDraw
import PIL.Image as Image

imagen = face_recognition.load_image_file("Marzo\image.png")
face_locations = face_recognition.face_locations(imagen)
print(face_locations)
face_landmarks_list = face_recognition.face_landmarks(imagen)
print(face_landmarks_list)
for face_landmarks in face_landmarks_list:
    pil_image = Image.fromarray(imagen)
    pil_image.show()
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
        d = ImageDraw.Draw(pil_image, 'RGBA')
        d.line(face_landmarks[facial_feature], fill=(255, 0, 0, 128), width=10)
    pil_image.show()