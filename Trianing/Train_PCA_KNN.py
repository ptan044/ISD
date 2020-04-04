from Image_tools import Face_recognition_tools as FRT
from Image_tools import Face_detection_tools as FDT
from sklearn import decomposition
from sklearn import neighbors, datasets

image_dict_list = FDT.load_images(FDT.BW_cropped_images_dir)
FRT.convert_all_to_vector(image_dict_list)
PCA_train_format = FRT.change_to_pca_format(image_dict_list)
pca = decomposition.PCA(n_components=80)

max_n_neighbors = 4
knn_list = []
pca.fit(PCA_train_format["Combined_array"])
PCA_train_format["Descriptive_vectors"] = pca.transform(PCA_train_format["Combined_array"])
for n_neighbors in range(max_n_neighbors):
    clf = neighbors.KNeighborsClassifier(n_neighbors + 1, weights='uniform')
    clf.fit(PCA_train_format["Descriptive_vectors"], PCA_train_format["Name_indexs"])
    knn_list.append(clf)


def recognise_face(img_arr, pca=pca, clf_list=knn_list, name_list=FRT.List_of_people, min_votes=3):
    img_vec = FRT.convert_to_vector(img_arr)
    descriptive_vectors = pca.transform([img_vec])
    votes = [0] * len(name_list)
    predictions = []
    for clf in clf_list:
        idx = int(clf.predict(descriptive_vectors))

        print(idx)
        predictions.append(name_list[idx])
        votes[idx] = votes[idx] + 1
    if max(votes) >= min_votes:
        persons_idx = votes.index(max(votes))
        person = name_list[persons_idx]

    else:
        person = "Unknown"
    return person, predictions


def recognise_image(img_arr, pca=pca, clf_list=knn_list, name_list=FRT.List_of_people, min_votes=3):
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
    bounding_boxes = FDT.get_face_bounding_boxes(img_arr)
    print(bounding_boxes)
    faces_detected = []
    for bounding_box in bounding_boxes:
        cropped_image = FDT.crop_image(img_arr, bounding_box)
        cropped_image = FDT.convert_to_BW_from_dict(cropped_image)

        final_label, predictions = recognise_face(img_arr, pca=pca, clf_list=clf_list, name_list=name_list,
                                                  min_votes=min_votes)
        face_dict = {'Cropped_image': cropped_image, 'Bounding_box': bounding_box, "Final_label": final_label,
                     'Predictions': predictions}
        faces_detected.append(face_dict)
    return faces_detected


# tester = image_dict_list[67]
# database1 = image_dict_list[10]
# print(FRT.get_distance(tester["Descriptive_vector"],database1["Descriptive_vector"]))
# names, distance  = FRT.rank_neighbours(tester,image_dict_list)
# print(names)
# print(distance)
#
# print(tester["Descriptive_vector"])
# print(clf.predict(tester["Descriptive_vector"]))

if __name__ == "__main__":
    test_image_name = "subject01_happy_0.jpg"
    test_image = FDT.load_image(FDT.test_dir, test_image_name)
    raw_test_name = "subject01_happy.jpg"
    raw_test_image = FDT.load_image(FDT.raw_images_dir, test_image_name)
    print(raw_test_image)

    print(recognise_face(test_image["Img_arr"]))
    print(recognise_image(raw_test_image['Img_arr']))


