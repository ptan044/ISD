import numpy as np
from sklearn import decomposition
from typing import List
from Image_tools import Face_detection_tools as FDT
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets

List_of_people = ["subject01", "subject02", "subject03", "subject04", "subject05",
                  "subject06", "subject07", "subject08", "subject09", "subject10",
                  "subject11", "subject12", "subject13", "subject14", "subject15",
                  "subject16", "subject17", "subject18", "subject19", "subject20",
                  "Kenneth", "Wang Han", "Pin Da"
                  ]


class Face_recog_image(FDT.Face_det_img):
    def __init__(self, img_arr):
        super().__init__(img_arr)
        self.img_vector = None


def change_to_pca_format(list_of_img: List[FDT.Database_face_det_img]):
    """ Converts a list of dictionaries to a dictionary of arrays for PCA form scikit learn
    Args:
        list_of_img:

    Returns:
        dictionary of
        {'Names':["Kenneth", "Zhi Wei", "Pin Da"],
         'Expression': ["Sad", "Happy", "Laughing"],
         'Combined_array':[]
    """
    name_list = []
    index_list = []
    expression_list = []
    combined_list = []
    for img in list_of_img:
        name_list.append(img.name)
        expression_list.append(img.expression)
        combined_list.append(img.img_vector.tolist())
        index_list.append(List_of_people.index(img.name))
    combined_arr = np.array(combined_list)

    return {'Names': name_list, 'Name_indexs': index_list, 'Expression': expression_list,
            'Combined_array': combined_arr}


def get_distance(descriptive_vector1, descriptive_vector2):
    """The distance between two descriptive vectors is found vector is found

    Args:
        descriptive_vector1:
        descriptive_vector2:

    Returns:
        float of distance between two vectors

    """
    return np.linalg.norm(descriptive_vector1 - descriptive_vector2)


def rank_neighbours(test_vector, comparison_database: List[FDT.Database_face_det_img]):
    names = []
    distances = []
    for image in comparison_database:

        comparison_name = image.name
        comparison_vector = image.descriptive_vector
        current_distance = get_distance(test_vector, comparison_vector)
        if len(names) == 0:
            distances.append(current_distance)
            names.append(comparison_name)
        else:
            if current_distance > distances[-1]:

                distances.append(current_distance)
                names.append(comparison_name)
            else:
                for idx, distance in enumerate(distances):
                    if current_distance < distance:
                        distances.insert(idx, current_distance)
                        names.insert(idx, comparison_name)
                        break

    return names, distances


def get_descriptors(list_of_dicts, pca):
    for img_dict in list_of_dicts:
        img_dict['Descriptive_vector'] = pca.transform([img_dict["Img_arr"]])
        img_dict['Idx'] = List_of_people.index(img_dict["Name"])

def convert_to_vector(img_arr,):
    if img_arr.ndim > 2:
        img_arr = img_arr[0:90, 0:90, 0]
    return img_arr.flatten()

if __name__ == "__main__":

    n_neighbors = 5
    test_array1 = np.ones([250, 250, 3])
    test_array2 = np.ones([250, 250, 3]) * 2
    test_array3 = np.ones([250, 250, 3]) * 3
    test_array4 = np.ones([250, 250, 3]) * 4
    test_dict_1 = {'Name': "Kenneth", 'Expression': "Sad", 'Img_arr': test_array1}
    test_dict_2 = {'Name': "Pin Da", 'Expression': "Sad", 'Img_arr': test_array2}
    test_dict_3 = {'Name': "Wang Han", 'Expression': "Sad", 'Img_arr': test_array3}
    list_of_dict = [test_dict_1, test_dict_2, test_dict_3]

    print(convert_all_to_vector(list_of_dict))
    PCA_train_format = change_to_pca_format(list_of_dict)
    print(PCA_train_format)
    pca = decomposition.PCA(n_components=1)
    pca.fit(PCA_train_format["Combined_array"])
    PCA_train_format["Descriptive_vectors"] = pca.transform(PCA_train_format["Combined_array"])
    get_descriptors(list_of_dict, pca)
    print(list_of_dict)

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(PCA_train_format["Combined_array"], PCA_train_format["Name_indexs"])
    clf.predict()



