from Image_tools import Face_detection_tools as FDT


image_dict_list = FDT.load_images(FDT.raw_images_dir)
cropped_image_list = []

for image_dict in image_dict_list:
    FDT.get_bound_box_from_dict(image_dict)
    cropped_image_list = FDT.get_cropped_image_list_form_dict(image_dict)
    for idx, cropped_image in enumerate(cropped_image_list):
        cropped_image["Idx"] = str(idx)
        FDT.save_image_from_dict(FDT.cropped_images_dir,cropped_image)
    for idx, cropped_image in enumerate(cropped_image_list):
        FDT.convert_to_BW_from_dict(cropped_image)
        cropped_image["Idx"] = str(idx)
        FDT.save_image_from_dict(FDT.BW_cropped_images_dir, cropped_image)



