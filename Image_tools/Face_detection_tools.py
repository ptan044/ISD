import cv2
import argparse
import os, os.path
from math import sin, cos, radians
import numpy as np
from typing import List

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
    def __init__(self, center, radius, angle=None):
        self.radius = radius
        self.center = center
        self.angle = angle


class Rectangle:
    def __init__(self, coordiantes, angle=None):
        self.coordinates = coordiantes
        self.angle = angle


class Bounds:
    def __init__(self, rotated_box: Rectangle = None, rotated_circle: Circle = None, unrot_circle: Circle = None):
        self.rotated_box = rotated_box
        self.rotated_circle = rotated_circle
        self.unrot_circle = unrot_circle
        self.name = None
        self.img = None


class Face_det_img:
    def __init__(self, img_arr=None):

        self.img_arr = img_arr
        self.bound_list = []
        self.bounded_image = None
        self.img_vector = None
        if not self.img_arr is None:
            self.bounded_image = self.img_arr.copy()
        self.descriptive_vector = None

    def load(self, path, file_name):
        self.img_arr = cv2.imread(os.path.join(path, file_name))
        self.bounded_image = self.img_arr.copy()

    # region
    def generate_bounding_boxes_rotation_invariant(self, max_angle: int = 90, min_angle: int = -90, step_size=5):
        """
        Rotate within a range of values
        get bounding box for each rotation
        append bounding box and correspoding angle for each face detected

        Args:
            img_dict:

        Returns:

        """
        shape = self.img_arr.size
        for angle in range(min_angle, max_angle, step_size):
            rotated_img = self.__get_rotated_image(angle)
            bb = rotated_img.__get_face_bounding_boxes(shape)
            for coordinates in bb:
                bound = Bounds(Rectangle(coordinates, angle))
                self.bound_list.append(bound)
    def __rescale_face(self,face,shape,scale):

        x, y, w, h = face
        new_w = int(w/scale)
        new_h = int(h/scale)
        new_x = int(x/scale)
        new_y = int(y/scale)
        return new_x,new_y,new_w,new_h

    def __get_face_bounding_boxes(self,shape):
        """Uses Harr_like features to get bounding box for crop

        Args:
            image_arr: Array of face. Should be the output of loadImage

        Returns:
            Coordinates of the bounding box


        """
        reduction_factor = 0.4
        low_dim_image = cv2.resize(self.img_arr,(0,0),fx =reduction_factor, fy = reduction_factor)
        faces = face_cascade.detectMultiScale(low_dim_image, 1.3, 5)
        for idx, face in enumerate(faces):
            faces[idx] = self.__rescale_face(face, shape, reduction_factor)
        return faces

    def __get_rotated_image(self, angle):
        if not angle == 0:
            height, width = self.img_arr.shape[:2]

            rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            img_arr = cv2.warpAffine(self.img_arr, rot_mat, (width, height), flags=cv2.INTER_LINEAR)

        else:
            img_arr = self.img_arr
        img = Face_det_img(img_arr=img_arr)
        return img

    def get_list_of_cropped_image(self):

        for idx, bounds in enumerate(self.bound_list):
            bounding_box = bounds.rotated_box
            rot_img = self.__get_rotated_image(bounding_box.angle)
            bounds.img = rot_img.__crop_image(bounding_box.coordinates).img_arr

    def generate_all_bounds_faster(self, max_angle: int = 90, min_angle: int = -90, step_size=5):

        shape = self.img_arr.shape
        for angle in range(min_angle, max_angle, step_size):
            rotated_img = self.__get_rotated_image(angle)
            bb = rotated_img.__get_face_bounding_boxes(shape)
            for coordinates in bb:
                rect = Rectangle(coordinates, angle)
                center, radius = self.__convert_box_to_circle(rect)
                rotated_circle = Circle(center, radius, angle)
                center, radius = self.__rotate_circle(center, radius, shape, angle)
                unrot_circle = Circle(center, radius)
                bound = Bounds(rect,rotated_circle,unrot_circle)
                self.bound_list.append(bound)


    def __crop_image(self, bounding_box):
        """Crops image based on bounding box

        Args:
            image_arr:
            bounding_box:

        Returns:
            image array


        """

        x, y, w, h = bounding_box
        cropped_arr = self.img_arr[y:y + h, x:x + w]  # cropped faces
        cropped = Face_det_img(img_arr=cropped_arr)
        cropped.__ResizeWithAspectRatio(width=100, height=100)
        return cropped

    def __ResizeWithAspectRatio(self, width, height, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = self.img_arr.shape[:2]

        if width is None and height is None:
            pass

        elif width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        self.img_arr = cv2.resize(self.img_arr, dim, interpolation=inter)
        if not (self.img_arr.shape[0] == height and self.img_arr.shape[1] == width):
            print("padding reuiqred")
            delta_w = width - self.img_arr.shape[1]
            delta_h = height - self.img_arr.shape[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [255, 0, 0]
            self.img_arr = cv2.copyMakeBorder(self.img_arr, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)

    def convert_to_BW(self):
        self.img_arr = cv2.cvtColor(self.img_arr, cv2.COLOR_BGR2GRAY)
    def normalise(self):
        self.img_arr = cv2.normalize(self.img_arr, None, alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

    def remove_repeated_cricles(self, circle_centre_thrshold=0.3, radius_difference_threshold=0.3,
                                group_acceptance_threshold=0.9):
        bound_groups = []

        for bounds in self.bound_list:

            free_circle = bounds.unrot_circle
            if len(bound_groups) == 0:
                bound_groups.append([bounds])
            else:
                group_found = False
                for bound_group in bound_groups:

                    in_group = 0

                    for grouped_bound in bound_group:
                        if self.__if_close_circle(free_circle, grouped_bound.unrot_circle, circle_centre_thrshold,
                                                  radius_difference_threshold):
                            if self.__if_close_circle(grouped_bound.unrot_circle, free_circle, circle_centre_thrshold,
                                                      radius_difference_threshold):
                                in_group += 1

                    if in_group > group_acceptance_threshold * len(bound_group):
                        bound_group.append(bounds)
                        group_found = True
                        break;
                if not group_found:
                    bound_groups.append([bounds])

        self.bound_list = []
        for bound_group in bound_groups:
            selected_bound = bound_group[0]
            self.bound_list.append(selected_bound)

    def __if_close_circle(self, test_circle: Circle, label_circle: Circle, center_closeness_threshold,
                          radius_closeness_threshold):
        """
        Test if test circle is inside target circle
        Args:
            test_circle:
            label_circle:

        Returns:

        """
        center_is_close = False
        similar_radius = False
        cen_dist = np.linalg.norm(np.array(test_circle.center) - np.array(label_circle.center))
        if cen_dist < center_closeness_threshold * label_circle.radius:
            center_is_close = True

        rad_diff = test_circle.radius - label_circle.radius
        rad_diff = np.abs(rad_diff)
        if rad_diff < radius_closeness_threshold * label_circle.radius:
            similar_radius = True
        return similar_radius and center_is_close

    def draw_circles(self, circle: Circle):

        if self.bounded_image is None:
            self.bounded_image = self.img_arr.copy()

        cv2.circle(self.bounded_image, circle.center, circle.radius, (255, 0, 0))

    def __convert_box_to_circle(self, boundingbox: Rectangle):
        x, y, w, h = boundingbox.coordinates
        center = int(x + w / 2), int(y + h / 2)
        radius = int(((w / 2) ** 2 + (h / 2) ** 2) ** 0.5)
        return center, radius

    def convert_rotbb_to_circle(self):

        for bounds in self.bound_list:
            center, radius = self.__convert_box_to_circle(bounds.rotated_box)
            bounds.rotated_circle = Circle(center, radius, bounds.rotated_box.angle)

    def draw_bounding_box(self, bounding_box: Rectangle):
        x, y, w, h = bounding_box.coordinates
        cv2.rectangle(self.img_arr, (x, y), (x + w, y + h), (255, 0, 0))

    def un_rotate_circles(self):

        shape = self.img_arr.shape
        for bounds in self.bound_list:
            bounding_circle = bounds.rotated_circle
            center, radius = self.__rotate_circle(bounding_circle.center, bounding_circle.radius, shape,
                                                  bounding_circle.angle)
            unrot_circle = Circle(center, radius)
            bounds.unrot_circle = unrot_circle

    def __rotate_circle(self, center, radius, shape, angle):
        y_offset = shape[0]
        x_offset = shape[1]
        y_offset = y_offset / 2
        x_offset = x_offset / 2
        x, y = center
        x = x - x_offset
        y = y - y_offset
        # print("before rotate {}{}".format(x, y))
        Tx = x * cos(radians(angle)) - y * sin(radians(angle))
        Ty = x * sin(radians(angle)) + y * cos(radians(angle))
        # print("after rotate {}{}".format(Tx, Ty))
        x = Tx + x_offset
        y = Ty + y_offset
        center = (int(x), int(y))
        return center, radius
    def convert_to_vector(self):
        img = self.img_arr[0::, 0::, 0]
        self.img_vector =img.flatten()


class Database_face_det_img(Face_det_img):
    def __init__(self, img_arr=None):
        super().__init__(img_arr=img_arr)
        self.name = None
        self.expression = None

    def load(self, path, file_name):
        """
            Args:
                path: The folder which the file is found
                file_name: The name of the file, in format <Name>_<Expression>.jpeg

            Returns:
                A dictionary of the following for example:
                {'Name': "Kenneth", 'Expression': "Sad", "Img_arr": image_arr}

            """
        super().load(path, file_name)
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

        file_name = self.name + "_" + self.expression + ".jpg"
        save_path = os.path.join(path, file_name)
        print("saving {}".format(save_path))

        cv2.imwrite(save_path, self.img_arr)


        # if not self.bounded_image is None:
        #     file_name = self.name + "_" + self.expression + "bounded.jpg"
        #     save_path = os.path.join(path, file_name)
        #     cv2.imwrite(save_path, self.bounded_image)

    def get_list_of_cropped_data_base_image(self):
        data_base_image_list = []
        super().get_list_of_cropped_image()
        for idx, bounds in enumerate(self.bound_list):
            data_base_image = Database_face_det_img(bounds.img)
            data_base_image.name = self.name
            data_base_image.expression = self.expression + str(idx)
            data_base_image_list.append(data_base_image)
        return data_base_image_list


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
        img = Database_face_det_img()
        img.load(path,file)
        output.append(img)
    return output


if __name__ == "__main__":
    images = load_images(raw_images_dir)
    for image in images:
        # image.convert_to_BW()
        image.generate_bounding_boxes_rotation_invariant(1,0,10)
        image.convert_rotbb_to_circle()
        image.un_rotate_circles()
        image.remove_repeated_cricles()
        cropped_images_list = image.get_list_of_cropped_data_base_image()

        for cropped_image in cropped_images_list:
            cropped_image.convert_to_BW()
            cropped_image.normalise()
            cropped_image.save_data_base_image(cropped_images_dir)

    # test_image_name_1 = "Pin Da_test.jpg"
    # test_image_1 = os.path.join(raw_images_dir, test_image_name_1)
    #
    # test_img = Database_face_det_img()
    # print("Loading....")
    # test_img.load(raw_images_dir, test_image_name_1)
    # print("Loaded....")
    # # test_img.convert_to_BW()
    # print("getting bounds...")
    # test_img.generate_bounding_boxes_rotation_invariant(60, -60, 15)
    # print("bounds obtained")
    # test_img.convert_rotbb_to_circle()
    # test_img.un_rotate_circles()
    #
    # test_img.remove_repeated_cricles()
    # for bounds in test_img.bound_list:
    #     test_img.draw_circles(bounds.unrot_circle)
    # test_img.save_data_base_image(test_dir)



