from __future__ import print_function
import cv2
import argparse
import os, os.path

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade




# Function to resize input images



imageDir = "D:\\Face Database"  # specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg"]  # specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]



for imagePath in image_path_list:
    img = cv2.imread(imagePath)
    if img is None:
        continue

    resized = ResizeWithAspectRatio(img, width=250)

    faces = face_cascade.detectMultiScale(resized, 1.3, 5)  # detect faces within resized image
    for (x, y, w, h) in faces:
        resized = cv2.rectangle(resized, (x, y), (x + w, y + h + 10), (255, 255, 255),
                                2)  # resized image with face detected
        cropped = resized[y:y + h, x:x + w]  # cropped faces
        recropped = ResizeWithAspectRatio(cropped, width=250, height=250)
        savePath = "D:\\Cropped Face Database"
        cv2.imwrite(os.path.join(savePath, 'image_' + str(num) + '.jpg'), recropped)

        num += 1

    # cv2.imshow(imagePath, resized)
    # cv2.imshow(imagePath, cropped)
    # cv2.imshow(imagePath, recropped)

cv2.destroyAllWindows()

# img = cv2.imread('D:\\Face Database\\subject01_happy.jpg')             # Read original image
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# resize = ResizeWithAspectRatio(img, width=250)          # Resize by width OR
# resize = ResizeWithAspectRatio(image, height=300)     # Resize by height

# faces = face_cascade.detectMultiScale(resize, 1.3, 5) # detect faces within resized image
# for (x,y,w,h) in faces:
#    resize = cv2.rectangle(resize,(x,y),(x+w,y+h+10),(255,0,0),2)
# roi_gray = gray[y:y+h, x:x+w]
#    roi_color = resize[y:y+h, x:x+w]

# cv2.imshow('Image', resize) # show resized image
# cv2.imshow('Detected', resize) # show face detected within resized image

# cv2.waitKey(0)

# camera_device = args.camera
# -- 2. Read the video stream
# cap = cv2.VideoCapture(camera_device)
# if not cap.isOpened:
#    print('--(!)Error opening video capture')
#    exit(0)
# while True:
#    ret, frame = cap.read()
#    if frame is None:
#        print('--(!) No captured frame -- Break!')
#        break
#    detectAndDisplay(frame)
#    if cv2.waitKey(10) == 27:
#        break