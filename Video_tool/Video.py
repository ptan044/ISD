import cv2
import Image_tools.Face_detection_tools as FDT
import Trianing.Train_PCA_KNN as PCA_KNN
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)
lineType = 2
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()




    # Our operations on the frame come here
    img = FDT.Face_det_img(frame)
    img.generate_all_bounds_faster(180,-180,40)
    img.remove_repeated_cricles()
    PCA_KNN.recognise_image(img)


    for bounds in img.bound_list:
        img.draw_circles(bounds.unrot_circle)
        bottom_left_corner = bounds.unrot_circle.center
        cv2.putText(img.bounded_image,bounds.name, bottom_left_corner, font, fontScale, fontColor, lineType)


    # Display the resulting frame
    display_image = img.bounded_image
    cv2.imshow('frame',display_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()