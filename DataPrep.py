import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

DATADIR = r"C:\Users\Nelson\Desktop\year 4 sem 2\EE4208\assignment\FaceData\Cropped Face Database"
CATEGORIES = ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11'
            , 'subject_12', 'subject_13', 'subject_14', 'subject_15', 'subject_16', 'subject_17', 'subject_18', 'subject_19', 'angry', 'happy', 'sad', 'sleepy', 'surprised'] 

# y_train = ['subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11'
#             , 'subject_12', 'subject_13', 'subject_14', 'subject_15', 'subject_16', 'subject_17', 'subject_18', 'subject_19', 'angry', 'happy', 'sad', 'sleepy', 'surprise']

training=[]
class_num=24

x_data = []
y_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to cats or dogs directory
        if category == 'subject_1':
            class_num = 1
        if category == 'subject_2':
            class_num = 2
        if category == 'subject_3':
            class_num = 3
        if category == 'subject_4':
            class_num = 4
        if category == 'subject_5':
            class_num = 5    
        if category == 'subject_6':
            class_num = 6    
        if category == 'subject_7':
            class_num = 7    
        if category == 'subject_8':
            class_num = 8
        if category == 'subject_9':
            class_num = 9
        if category == 'subject_10':
            class_num = 10
        if category == 'subject_11':
            class_num = 11
        if category == 'subject_12':
            class_num = 12
        if category == 'subject_13':
            class_num = 13    
        if category == 'subject_14':
            class_num = 14
        if category == 'subject_15':
            class_num = 15
        if category == 'subject_16':
            class_num = 16
        if category == 'subject_17':
            class_num = 17
        if category == 'subject_18':
            class_num = 18
        if category == 'subject_19':
            class_num = 19
        if category == 'happy':
            class_num = 20    
        if category == 'sad':
            class_num = 21
        if category == 'angry':
            class_num = 22
        if category == 'sleepy':
            class_num = 23
        if category == 'surprised':
            class_num = 24

        for img in os.listdir(path): #reiterate inside folder
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) #convert image to array and greyscale it, remove cv2.IMREAD_GRAYSCALE to get original image
                x_data.append(img_array)
                y_data.append(class_num)
            except Exception as e:
                pass
            
create_training_data()

fi=open('x_data.txt','w')
for i in range(len(x_data)):
    sti=str(x_data[i])
    fi.write(sti+'\n')
fi.close()

f=open('y_data.txt','w')
for i in range(len(y_data)):
    sti=str(y_data[i])
    f.write(sti+'\n')
f.close()

#print(x_data)

# print(x_data[0])
# print(y_data[0])
# print(np.shape(x_data))
# print(np.shape(y_data))