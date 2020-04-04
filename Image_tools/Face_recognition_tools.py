import numpy as np
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets
List_of_people = ["subject01", "subject02", "subject03", "subject04", "subject05",
                  "subject06", "subject07", "subject08", "subject09", "subject10",
                  "subject11", "subject12", "subject13", "subject14", "subject15",
                  "subject16", "subject17", "subject18", "subject19", "subject20",
                  "Kenneth", "Wang Han", "Pin Da"
                  ]


def convert_all_to_vector(list_of_dicts):
    """

    Args:
        list: List of dictionary whereby each dictionary is
        {'Name': "Kenneth", 'Expression': "Sad" 'Img_arr': image_arr}

    Returns:

    """
    for img_dict in list_of_dicts:
        img_dict["Img_arr"] = convert_to_vector(img_dict["Img_arr"])
    return  list_of_dicts

def convert_to_vector(img_arr):
    """converts a Black n White image array to a column vector (array)

    Args:
        image_arr:

    Returns:


    """
    img = img_arr[0:248, 0:248, 0]
    img = img.flatten()
    return img

def change_to_pca_format(list_of_dict):
    """ Converts a list of dictionaries to a dictionary of arrays for PCA form scikit learn
    Args:
        list_of_dict:

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
    for img_dict in list_of_dict:
        name_list.append(img_dict["Name"])
        expression_list.append(img_dict["Expression"])
        combined_list.append(img_dict["Img_arr"].tolist())
        index_list.append(List_of_people.index(img_dict["Name"]))
    combined_arr = np.array(combined_list)
    return {'Names': name_list, 'Name_indexs': index_list, 'Expression': expression_list, 'Combined_array': combined_arr}

def get_distance(descriptive_vector1, descriptive_vector2 ):
    """The distance between two descriptive vectors is found vector is found

    Args:
        descriptive_vector1:
        descriptive_vector2:

    Returns:
        float of distance between two vectors

    """
    return np.linalg.norm(descriptive_vector1 - descriptive_vector2)

def rank_neighbours(test_img_dict,comparison_database):
    names = []
    distances = []
    for image_dict in comparison_database:

        test_vector = test_img_dict["Descriptive_vector"]
        comparison_name = image_dict["Name"]
        comparison_vector = image_dict["Descriptive_vector"]
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
                        distances.insert(idx,current_distance)
                        names.insert(idx,comparison_name)
                        break

    return names, distances

def get_descriptors(list_of_dicts, pca):
    for img_dict in list_of_dicts:
        img_dict['Descriptive_vector'] = pca.transform([img_dict["Img_arr"]])
        img_dict['Idx'] = List_of_people.index(img_dict["Name"])



if __name__ == "__main__":
    n_neighbors = 5
    test_array1 = np.ones([250,250,3])
    test_array2 = np.ones([250, 250, 3])*2
    test_array3 = np.ones([250, 250, 3]) * 3
    test_array4 = np.ones([250, 250, 3]) * 4
    test_dict_1 = {'Name': "Kenneth", 'Expression': "Sad", 'Img_arr': test_array1}
    test_dict_2 = {'Name': "Pin Da", 'Expression': "Sad", 'Img_arr': test_array2}
    test_dict_3 = {'Name': "Wang Han", 'Expression': "Sad", 'Img_arr': test_array3}
    list_of_dict = [test_dict_1,test_dict_2, test_dict_3]
    print(list_of_dict)
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


    # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(PCA_train_format["Descriptive_vectors"])
    # target =  [list_of_dict[0]["Img_arr"]]
    # print(target)
    # distances, indices = nbrs.kneighbors(target)
    # print(distances)
    # print(indices)
    #
