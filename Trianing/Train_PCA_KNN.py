from Image_tools import Face_recognition_tools as FRT
from Image_tools import Face_detection_tools as FDT
from sklearn import decomposition
from sklearn import neighbors, datasets


# region Loading data for PCA
image_list = FDT.load_images(FDT.BW_cropped_images_dir)
# endregion
# region converting data format to train PCA
for img in image_list:
    img.convert_to_vector()
PCA_train_format = FRT.change_to_pca_format(image_list)
# endregion
# region Training PCA
pca = decomposition.PCA(n_components=80)

pca.fit(PCA_train_format["Combined_array"])
# endregion



# converting images to descriptive vectors
PCA_train_format["Descriptive_vectors"] = pca.transform(PCA_train_format["Combined_array"])
for img in image_list:
    img.descriptive_vector = pca.transform([img.img_vector])
    # print(img.descriptive_vector)
# region training multiple KNN models with diffferent n
max_n_neighbors = 4
knn_list = []
for n_neighbors in range(max_n_neighbors):
    clf = neighbors.KNeighborsClassifier(n_neighbors + 1, weights='uniform')
    clf.fit(PCA_train_format["Descriptive_vectors"], PCA_train_format["Name_indexs"])
    knn_list.append(clf)
# endregion


def recognise_face(img_arr, pca=pca, clf_list=knn_list, name_list=FRT.List_of_people, min_votes=3):

    img_vec = FRT.convert_to_vector(img_arr)
    descriptive_vectors = pca.transform([img_vec])
    votes = [0] * len(name_list)
    predictions = []
    for clf in clf_list:
        idx = int(clf.predict(descriptive_vectors))

        predictions.append(name_list[idx])
        votes[idx] = votes[idx] + 1
    names, distance = FRT.rank_neighbours(descriptive_vectors,image_list)
    print(names)
    print(distance)
    if distance[0] > 4500:
        person = "Unknown"
    elif max(votes) >= min_votes:
        persons_idx = votes.index(max(votes))
        person = name_list[persons_idx]
    else:
        person = "Unknown"
    print(predictions)
    return person, predictions

def recognise_image(img: FDT.Face_det_img, pca=pca, clf_list=knn_list, name_list=FRT.List_of_people, min_votes=3):
    """
    converts to BW
    get list of bounding boxes for faces
    crops faces
    run PCA
    runs classification
    Args:
        img_arr:
        pca:
        clf:

    Returns:

    """
    img.convert_to_BW()
    img.generate_bounding_boxes_rotation_invariant(20,-20,15)
    img.generate_bounding_boxes_rotation_invariant(1, 0, 10)
    img.convert_rotbb_to_circle()
    img.un_rotate_circles()
    img.remove_repeated_cricles()
    img.get_list_of_cropped_image()
    for bound in img.bound_list:

        final_label, predictions = recognise_face(bound.img, pca=pca, clf_list=clf_list, name_list=name_list,
                                                  min_votes=min_votes)
        bound.name = final_label



if __name__ == "__main__":
    test_image_name = "subject01_happy.jpg"
    test_image = FDT.Face_det_img()
    test_image.load(FDT.test_dir, test_image_name)
    print(test_image.img_arr)
    print(recognise_image(test_image))



