import cv2


def resize_image(img, shape=(100, 100)):
    return cv2.resize(img, shape)



def crop_face(img):
    cascade = cv2.CascadeClassifier("../../data/haarcascades/haarcascade_frontalface_default.xml")
    # print(img)
    faces = cascade.detectMultiScale(img)
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
        return sub_face
