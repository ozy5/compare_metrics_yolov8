import os
from PIL import Image
import numpy as np
import glob
from ultralytics import YOLO
import cv2
import utils.utils as utils

SAVE_PATH_ROOT = "/home/umut/Desktop/TEST_EXPERIMENTS/OUTPUTS"

SELECTED_IMAGES_PATH = "/home/umut/Desktop/TEST_EXPERIMENTS/SELECTED_IMAGES_TO_PAPER"

#filtered
EXP_PATH_1 = "/home/umut/Desktop/TEST_EXPERIMENTS/hist_eq_filtered_variations/175_100/datasets"

EXP1_PREFIX = "filtered_dataset_"


#not filtered
EXP_PATH_2 = "/home/umut/Desktop/TEST_EXPERIMENTS/hist_eq_filtered_variations/not_filtered/datasets"

EXP2_PREFIX = "not_filtered_dataset_"

for dataset_name in os.listdir(SELECTED_IMAGES_PATH):

    dataset_path = os.path.join(SELECTED_IMAGES_PATH, dataset_name)

    dataset_num = dataset_name[-1]

    filtered_exp_path = os.path.join(EXP_PATH_1, (EXP1_PREFIX + dataset_num))

    not_filtered_exp_path = os.path.join(EXP_PATH_2, (EXP2_PREFIX + dataset_num))

    for image_name in os.listdir(dataset_path):

        image_name_without_ext = image_name.split(".")[0]


        images_filtered_root_path = os.path.join(filtered_exp_path, "test", "images")

        images_not_filtered_root_path = os.path.join(not_filtered_exp_path, "test", "images")

        weights_filtered_path = os.path.join(filtered_exp_path, "best.pt")

        weights_not_filtered_path = os.path.join(not_filtered_exp_path, "best.pt")

        EXP_NUM = filtered_exp_path[-1]

        labels_filtered_path = os.path.join(filtered_exp_path, "test", "labels")
        labels_not_filtered_path = os.path.join(not_filtered_exp_path, "test", "labels")



        # THE NUMBER -3 IS PATH SPECIFIC
        weights_filtered_name = weights_filtered_path.split("/")[-1]
        weights_not_filtered_name = weights_not_filtered_path.split("/")[-1]


        SAVE_PATH = os.path.join(SAVE_PATH_ROOT, ("dataset_" + str(EXP_NUM)))

        os.makedirs(SAVE_PATH, exist_ok=True)



        CONF_THRES = 0.2
        IOU_THRES = 0.001
        LINE_THICKNESS = 2


        model_filtered = YOLO(weights_filtered_path)
        model_not_filtered = YOLO(weights_not_filtered_path)



        images_filtered= glob.glob(os.path.join(images_filtered_root_path, "*"))
        images_not_filtered= glob.glob(os.path.join(images_not_filtered_root_path, "*"))


        filtered_img = os.path.join(images_filtered_root_path, image_name)
        not_filtered_img = os.path.join(images_not_filtered_root_path, image_name)
    
        filtered_img_name = filtered_img.split("/")[-1]
        not_filtered_img_name = not_filtered_img.split("/")[-1]

        if(filtered_img_name != not_filtered_img_name):
            print("Not equal datasets by image names")
            exit()

        #get the results
        filtered_img_result = model_filtered.predict(task="detect", source=filtered_img, show_labels=False)[0]
        not_filtered_img_result = model_not_filtered.predict(task="detect", source=not_filtered_img, show_labels=False)[0]

        # #plot the bboxes
        # annotated_filtered_img = filtered_img_results.plot(line_width = LINE_THICKNESS, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
        # annotated_not_filtered_img = not_filtered_img_results.plot(line_width = LINE_THICKNESS, conf_thres=CONF_THRES, iou_thres=IOU_THRES)


        #plot the bbox rectangles manually
        annotated_filtered_img = filtered_img_result.orig_img
        annotated_not_filtered_img = not_filtered_img_result.orig_img

        #make a deep copy to only apply the ground truth bboxes
        annotated_filtered_img_gt = np.array(annotated_filtered_img)
        annotated_not_filtered_img_gt = np.array(annotated_not_filtered_img)

        #save original images with imageio
        #turn BGR to RGB PIL image
        original_filtered_to_save = Image.fromarray(cv2.cvtColor(annotated_filtered_img, cv2.COLOR_BGR2RGB))
        original_not_filtered_to_save = Image.fromarray(cv2.cvtColor(annotated_not_filtered_img, cv2.COLOR_BGR2RGB))

        #save the images
        original_filtered_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_original_filtered.png")))
        original_not_filtered_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_original_not_filtered.png")))

        filtered_height, filtered_width = filtered_img_result.orig_shape
        not_filtered_height, not_filtered_width = not_filtered_img_result.orig_shape

        filtered_bboxes = filtered_img_result.boxes.xyxy
        not_filtered_bboxes = not_filtered_img_result.boxes.xyxy

        for bbox in filtered_bboxes:
            annotated_filtered_img = cv2.rectangle(annotated_filtered_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), LINE_THICKNESS)

        for bbox in not_filtered_bboxes:
            annotated_not_filtered_img = cv2.rectangle(annotated_not_filtered_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), LINE_THICKNESS)


        #save the annotated images with imageio
        #turn BGR to RGB PIL image
        annotated_filtered_to_save = Image.fromarray(cv2.cvtColor(annotated_filtered_img, cv2.COLOR_BGR2RGB))
        annotated_not_filtered_to_save = Image.fromarray(cv2.cvtColor(annotated_not_filtered_img, cv2.COLOR_BGR2RGB))

        #save the images
        annotated_filtered_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_annotated_filtered.png")))
        annotated_not_filtered_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_annotated_not_filtered.png")))




        #plot the ground truth bboxes in blue color
        
        #get the ground truth bboxes for filtered
        with open(os.path.join(labels_filtered_path, filtered_img_name.split(".")[0] + ".txt"), "r") as f:
            filtered_gt_bboxes = f.readlines()

            #split the bboxes
            filtered_gt_bboxes = [[float(bbox_val) for bbox_val in bbox.strip().split(" ")[1:5]] for bbox in filtered_gt_bboxes]
            
            #convert the bboxes to xyxy format
            filtered_gt_bboxes = [utils.xywh_2_xyxy(bbox) for bbox in filtered_gt_bboxes]

            #convert the bboxes to actual pixel values
            filtered_gt_bboxes = [utils.normalized_2_actual_pixel_values(bbox, filtered_width, filtered_height) for bbox in filtered_gt_bboxes]

            #plot the bboxes
            for bbox in filtered_gt_bboxes:
                annotated_filtered_img = cv2.rectangle(annotated_filtered_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)

                annotated_filtered_img_gt = cv2.rectangle(annotated_filtered_img_gt, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
        
        #get the ground truth bboxes for not filtered
        with open(os.path.join(labels_not_filtered_path, not_filtered_img_name.split(".")[0] + ".txt"), "r") as f:
            not_filtered_gt_bboxes = f.readlines()

            #split the bboxes
            not_filtered_gt_bboxes = [[float(bbox_val) for bbox_val in bbox.strip().split(" ")[1:5]] for bbox in not_filtered_gt_bboxes]
            
            #convert the bboxes to xyxy format
            not_filtered_gt_bboxes = [utils.xywh_2_xyxy(bbox) for bbox in not_filtered_gt_bboxes]

            #convert the bboxes to actual pixel values
            not_filtered_gt_bboxes = [utils.normalized_2_actual_pixel_values(bbox, not_filtered_width, not_filtered_height) for bbox in not_filtered_gt_bboxes]

            #plot the bboxes
            for bbox in not_filtered_gt_bboxes:
                annotated_not_filtered_img = cv2.rectangle(annotated_not_filtered_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)

                annotated_not_filtered_img_gt = cv2.rectangle(annotated_not_filtered_img_gt, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)

        #save the annotated images with ground truths with imageio
        #turn BGR to RGB PIL image
        annotated_filtered_gt_to_save = Image.fromarray(cv2.cvtColor(annotated_filtered_img, cv2.COLOR_BGR2RGB))
        annotated_not_filtered_gt_to_save = Image.fromarray(cv2.cvtColor(annotated_not_filtered_img, cv2.COLOR_BGR2RGB))

        #save the images
        annotated_filtered_gt_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_gt_with_annotations_filtered.png")))
        annotated_not_filtered_gt_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_gt_with_annotations_not_filtered.png")))

        #save the only ground truth bboxes with imageio
        #turn BGR to RGB PIL image
        filtered_gt_to_save = Image.fromarray(cv2.cvtColor(annotated_filtered_img_gt, cv2.COLOR_BGR2RGB))
        not_filtered_gt_to_save = Image.fromarray(cv2.cvtColor(annotated_not_filtered_img_gt, cv2.COLOR_BGR2RGB))

        #save the images
        filtered_gt_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_only_gt_filtered.png")))
        not_filtered_gt_to_save.save(os.path.join(SAVE_PATH, (image_name_without_ext + "_only_gt_not_filtered.png")))


        # #concatenate images horizontally
        # concatenated_img = np.concatenate((annotated_filtered_img, annotated_not_filtered_img), axis=1)

        # #turn BGR to RGB PIL image
        # concatenated_img = Image.fromarray(cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB))

        # #save the concatenated image
        # concatenated_img.save(os.path.join(SAVE_PATH, filtered_img_name))

            
