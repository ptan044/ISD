import cv2
import argparse
import os, os.path
from math import sin, cos, radians
import numpy as np
current_dir = os.getcwd()
master_dir = current_dir.split("\src")[0]
data_dir = os.path.join(master_dir, "data")
raw_images_dir = os.path.join(data_dir, "raw")
cropped_images_dir = os.path.join(data_dir, "cropped")
BW_cropped_images_dir = os.path.join(data_dir, "BW_cropped")
test_dir = os.path.join(data_dir, "test")
rotation_directory = os.path.join(data_dir, "rotation_test")
haar_cascade = os.path.join(data_dir, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(haar_cascade):
    print('--(!)Error loading face cascade')
    exit(0)
class Circle:
    def __init__(self,center, radius, angle = None):
        self.radius = radius
        self.center = center
        self.angle = angle

class Rectangle:
    def __init__(self, coordiantes, angle):
        self.coordinates = coordiantes
        self.angle = angle

class Bounds:
    def __init__(self, rotated_box:Rectangle, rotated_circle: Circle, unrot_circle: Circle):
        self.rotated_box = rotated_box
        self.rotated_circle = rotated_circle
        self.unrot_cicles = unrot_circle

class Face_det_img:
    def __init__(self):

        self.img_arr = None
        self.bound = []
    def load(self,path, file_name):
        self.img_arr = cv2.imread(os.path.join(path, file_name))

class Database_face_det_img(Face_det_img):
    def __init__(self):
        super().__init__()
        self.name = None
        self.expression = None
    def load(self,path,file_name):
        """
            Args:
                path: The folder which the file is found
                file_name: The name of the file, in format <Name>_<Expression>.jpeg

            Returns:
                A dictionary of the following for example:
                {'Name': "Kenneth", 'Expression': "Sad", "Img_arr": image_arr}

            """
        details = file_name.split(".")[0]
        self.name = details.split("_")[0]
        self.expression = details.split("_")[1]

    def save_data_base_image(self, path):
        """

        Args:
            path:
            file_name:
            image_arr:

        Returns:

        """
        file_name = self.
        save_path = os.path.join(path, file_name)
        cv2.imwrite(save_path, image_arr)


def rotate_image(img, angle):
    if not angle == 0:
        height, width = img.shape[:2]

        rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img = cv2.warpAffine(img, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return img


def rotate_point(boundingbox, img, angle):
    if angle == 0: return boundingbox
    x = boundingbox[0] - img.shape[1] * 0.4
    y = boundingbox[1] - img.shape[0] * 0.4
    newx = x * cos(radians(angle)) - y * sin(radians(angle)) + img.shape[1] * 0.5
    newy = x * sin(radians(angle)) + y * cos(radians(angle)) + img.shape[0] * 0.5
    return int(newx), int(newy), boundingbox[2], boundingbox[3]


def load_images(path):
    """

    Args:
        path:

    Returns:
        List of dictionary whereby each dictionary is
        {'Name': "Kenneth", 'Expression': "Sad" 'Img_arr': image_arr}

    """
    valid_image_extensions = [".jpg", ".jpeg"]  # specify your valid extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]
    output = []
    for file in os.listdir(path):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_dict = load_image(path, file)
        output.append(image_dict)
    return output




def get_face_bounding_boxes(image_arr):
    """Uses Harr_like features to get bounding box for crop

    Args:
        image_arr: Array of face. Should be the output of loadImage

    Returns:
        Coordinates of the bounding box


    """
    faces = face_cascade.detectMultiScale(image_arr, 1.3, 5)
    return faces


def crop_image(image_arr, bounding_box):
    """Crops image based on bounding box

    Args:
        image_arr:
        bounding_box:

    Returns:
        image array


    """

    x, y, w, h = bounding_box
    resized = cv2.rectangle(image_arr, (x, y), (x + w, y + h + 10), (255, 255, 255),
                            2)  # resized image with face detected
    cropped = resized[y:y + h, x:x + w]  # cropped faces
    recropped = ResizeWithAspectRatio(cropped, width=100, height=100)

    return recropped


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def convert_to_BW(image_arr):
    """

    Args:
        image_arr:

    Returns:

    """
    return (cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY))


def get_bound_box_from_dict(img_dict):



def get_cropped_image_list_form_dict(img_dict):
    name = img_dict["Name"]
    expression = img_dict["Expression"]
    img_arr = img_dict["Img_arr"]
    bounding_boxes = img_dict["Bounding_box"]

    output_list = []
    for idx, bounding_box in enumerate(bounding_boxes):
        cropped_image_arr = crop_image(img_arr, bounding_box)
        cropped_img_dict = {"Name": name,
                            "Expression": expression,
                            "Img_arr": cropped_image_arr
                            }
        output_list.append(cropped_img_dict)
    return output_list


def save_image_from_dict(path, img_dict):
    if "Idx" in img_dict:
        file_name = img_dict["Name"] + "_" + img_dict["Expression"] + "_" + img_dict["Idx"] + ".jpg"
    else:
        file_name = img_dict["Name"] + "_" + img_dict["Expression"] + ".jpg"
    print("saving {} to {}...".format(file_name, path))
    save_image(path, file_name, img_dict["Img_arr"])


def convert_to_BW_from_dict(img_dict):
    img_dict["Img_arr"] = convert_to_BW(img_dict["Img_arr"])


def draw_rectangle_from_dict(img_dict, target_string):
    for x, y, w, h in img_dict[target_string]:
        cv2.rectangle(img_dict["Img_arr"], (x, y), (x + w, y + h), (255, 0, 0))


def get_bounding_boxes_rotation_invariant(img_dict):
    """
    Rotate within a range of values
    get bounding box for each rotation
    append bounding box and correspoding angle for each face detected

    Args:
        img_dict:

    Returns:

    """
    bounding_boxes = []
    for angle in range(-90, 90, 2):
        img_arr = rotate_image(img_dict["Img_arr"], angle)
        bb = get_face_bounding_boxes(img_arr)
        if len(bb):
            bounding_box = {"Bounding_box": bb, "Angle": angle}
            bounding_boxes.append(bounding_box)
    img_dict["Rot_bounding_boxes"] = bounding_boxes


def rotate_back_rectangles(img_dict):
    unrotated_bounding_boxes = []
    for bounding_box in img_dict["Rot_bounding_boxes"]:
        new_box = rotate_point(bounding_box["Bounding_box"][0], img_dict["Img_arr"], bounding_box["Angle"])
        unrotated_bounding_boxes.append(new_box)
    img_dict["Unrotated_bounding_box"] = unrotated_bounding_boxes


def convert_box_to_circle(boundingbox):
    x, y, w, h = boundingbox
    center = int(x + w / 2), int(y + h / 2)
    radius = int(((w / 2) ** 2 + (h / 2) ** 2) ** 0.5)
    return center, radius


def rotate_center(center, radius, shape, angle):
    y_offset, x_offset, _ = shape
    y_offset = y_offset / 2
    x_offset = x_offset / 2
    x, y = center
    x = x - x_offset
    y = y - y_offset
    print("before rotate {}{}".format(x, y))
    Tx = x * cos(radians(angle)) - y * sin(radians(angle))
    Ty = x * sin(radians(angle)) + y * cos(radians(angle))
    print("after rotate {}{}".format(Tx, Ty))
    x = Tx + x_offset
    y = Ty + y_offset
    center = (int(x), int(y))
    return center, radius


def convert_rotbb_to_circle(img_dict):
    bounding_circles = []
    for bb in img_dict["Rot_bounding_boxes"]:
        center, radius = convert_box_to_circle(bb["Bounding_box"][0])
        bc = {"Center": center, "Radius": radius, "Angle": bb["Angle"]}
        bounding_circles.append(bc)
    img_dict["Bounding_circles"] = bounding_circles


def un_rotate_circles(img_dict):
    unrotated_circles = []
    shape = img_dict["Img_arr"].shape
    for bc in img_dict["Bounding_circles"]:
        center, radius = rotate_center(bc["Center"], bc["Radius"], shape, bc["Angle"])
        bounding_circle = {"Center": center, "Radius": radius}
        unrotated_circles.append(bounding_circle)
    img_dict["unrot_bounding_circles"] = unrotated_circles
    img_dict["Bounding_circles"] = unrotated_circles


def draw_circles_from_dict(img_dict, target_label):
    for circle in img_dict[target_label]:
        cv2.circle(img_dict["Img_arr"], circle["Center"], circle["Radius"], (255, 0, 0))
def if_close_circle(test_circle, label_circle,center_closeness_thredhold, radius_closeness_threshold):
    """
    Test if test circle is inside target circle
    Args:
        test_circle:
        label_circle:

    Returns:

    """
    center_is_close = False
    similar_radius = False
    cen_dist = np.linalg.norm(np.array(test_circle['Center']) - np.array(label_circle['Center']))
    if cen_dist<center_closeness_thredhold*label_circle['Radius']:
        center_is_close = True

    rad_diff = test_circle['Radius'] - label_circle['Radius']
    rad_diff = np.abs(rad_diff)
    if rad_diff < radius_closeness_threshold*label_circle["Radius"]:
        similar_radius = True
    return similar_radius and center_is_close
def remove_repeated_cricles(img_dict, target_label, circle_centre_thrshold = 0.1, radius_difference_threshold = 0.1, group_acceptance_threshold = 0.9):
    circle_groups = []

    for free_circle in img_dict[target_label]:
        if len(circle_groups) == 0:
            circle_groups.append([free_circle])
        else:
            for circle_group in circle_groups:
                in_group = 0

                for grouped_circle in circle_group:
                    if if_close_circle(free_circle,grouped_circle,circle_centre_thrshold,radius_difference_threshold):
                        if if_close_circle(grouped_circle, free_circle,circle_centre_thrshold, radius_difference_threshold):
                            in_group += 1
                if in_group > group_acceptance_threshold*len(circle_group):
                    circle_group.append(free_circle)
    print(circle_groups)
    for circle_group in circle_groups:
        mean_radius = np.array([circle["Radius"] for circle in circle_group]).mean()
        print(mean_radius)





if __name__ == "__main__":

    test_image_name_1 = "Pin Da_Rotate2.jpg"
    test_image_1 = os.path.join(raw_images_dir, test_image_name_1)


    image1 = load_image(raw_images_dir, test_image_name_1)
    get_bounding_boxes_rotation_invariant(image1)
    convert_rotbb_to_circle(image1)
    un_rotate_circles(image1)
    draw_circles_from_dict(image1,"unrot_bounding_circles")
    remove_repeated_cricles(image1,"unrot_bounding_circles")
    save_image(rotation_directory, "detected_circles.jpg", image1["Img_arr"])

    test_circle = image1["unrot_bounding_circles"][0]
    label_circle = image1["unrot_bounding_circles"][1]
    print(if_close_circle(test_circle,label_circle,0.1,0.1))
