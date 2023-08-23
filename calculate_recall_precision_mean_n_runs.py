import os
import glob
import utils.utils as utils
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

CONF_THRESHOLD = 0.2

# IoU_threshold = 0.5
# IoU_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.9, 0.93, 0.96]
#IoU_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75]
IoU_thresholds = np.linspace(0.1, 0.52, 22, dtype=np.float32)

TRAININGS_PATH = "/home/umut/Desktop/TEST_EXPERIMENTS/TRAININGS"

TRAININGS = sorted(os.listdir(TRAININGS_PATH))


for training_name in TRAININGS:
    NAME = "_".join(training_name.split("_")[0:2])

    print(f"\nPROCESSING THE DATA PROM THE EXPERIMENT: {NAME}\n")


    DATASETS_PATH = os.path.join("/home/umut/Desktop/TEST_EXPERIMENTS/hist_eq_filtered_variations", NAME, "datasets")

    TEST_OR_VAL = "test"




    #create dictionary object to create pandas dataframe
    data = {
        "Dataset_Name" : [],
        "Test_or_Val": [],
        "IoU_Threshold": [],
        "Recall": [],
        "Precision": [],
        "F1_Score": [],
        "True_Positive_Count": [],
        "False_Positive_Count": [],
        "False_Negative_Count": []
    }

    #create the empty dataframe
    df = pd.DataFrame(data)


    dataset_names = sorted(os.listdir(DATASETS_PATH))

    dataset_count = len(dataset_names)

    for (dataset_num, current_dataset) in enumerate(dataset_names):
        
        #create dictionary object to create pandas dataframe for current dataset
        current_data = {
        "Dataset_Name" : [],
        "Test_or_Val": [],
        "IoU_Threshold": [],
        "Recall": [],
        "Precision": [],
        "F1_Score": [],
        "True_Positive_Count": [],
        "False_Positive_Count": [],
        "False_Negative_Count": []
        }


        #Determine the paths
        current_dataset_path = os.path.join(DATASETS_PATH, current_dataset)

        MODEL_PATH = os.path.join(current_dataset_path, "best.pt")

        IMAGES_PATH = os.path.join(current_dataset_path, TEST_OR_VAL, "images")

        LABELS_PATH = os.path.join(current_dataset_path, TEST_OR_VAL, "labels")

        #load the model
        model = YOLO(MODEL_PATH)

        #get the results
        results = model.predict(IMAGES_PATH, verbose=False, conf=CONF_THRESHOLD)

        # get prediction results
        predictions_xyxy_normalized = [(result.boxes.xyxyn.to("cpu")) for result in results]

        #get label paths
        label_paths = [os.path.join(LABELS_PATH, (os.path.splitext(os.path.basename(result.path))[0] + ".txt")) for result in results]


        #get label bboxes
        label_bboxes = [utils.get_xyxy_bboxes_from_YOLO_format_txt(label_path) for label_path in label_paths]


        for IoU_threshold in IoU_thresholds:

            #calculate the TP, FP and FN for images
            TP, FP, FN = utils.calculate_TP_FP_FN_all_images(predictions_xyxy_normalized, label_bboxes, IoU_threshold=IoU_threshold)

            #calculate the recall and precision for images
            recall, precision = utils.calculate_recall_and_precision_from_TP_FP_FN(TP, FP, FN)

            #calculate F1 score for images
            f1_score = utils.get_F1_score_from_recall_and_precision(recall, precision)

            
            # add values to current_data dictionary
            current_data["Dataset_Name"].append(current_dataset)
            current_data["Test_or_Val"].append(TEST_OR_VAL)
            current_data["IoU_Threshold"].append(IoU_threshold)
            current_data["Recall"].append(recall)
            current_data["Precision"].append(precision)
            current_data["F1_Score"].append(f1_score)
            current_data["True_Positive_Count"].append(TP)
            current_data["False_Positive_Count"].append(FP)
            current_data["False_Negative_Count"].append(FN)
            

        #create the dataframe for current dataset
        current_df = pd.DataFrame(current_data)
        
        #merge the current_data dictionary to data dictionary
        df = pd.concat([df, current_df], ignore_index=True)

        #print the current status
        print(f"{dataset_num+1}/{dataset_count} datasets are processed")



    # Get the mean of the dataframe, saving IoU threshold column deleting the Dataset_Name column
    df_mean = (df.drop(columns=["Dataset_Name", "Test_or_Val"])).groupby(["IoU_Threshold"]).mean().reset_index()


    #save the dataframe as csv
    df_mean.to_csv("csv_files/" + NAME + "_recall_precision_F1_score_TP_FP_FN.csv", index=False)

    ## VISIULATION PART

    #plot 6 graphs as 2x3 for recall, precision and F1 score as subplots, TP, FP, FN. Also name the x and y axes and add title
    fig, axs = plt.subplots(2,3, figsize=(20, 10))
    fig.suptitle('Recall, Precision, F1 Score, TP, FP, FN vs IoU Threshold')

    axs[0,0].plot(df_mean["IoU_Threshold"], df_mean["Recall"])
    axs[0,0].set_title("Recall")
    axs[0,0].set_xlabel("IoU Threshold")
    axs[0,0].set_ylabel("Recall")

    axs[0,1].plot(df_mean["IoU_Threshold"], df_mean["Precision"])
    axs[0,1].set_title("Precision")
    axs[0,1].set_xlabel("IoU Threshold")
    axs[0,1].set_ylabel("Precision")

    axs[0,2].plot(df_mean["IoU_Threshold"], df_mean["F1_Score"])
    axs[0,2].set_title("F1 Score")
    axs[0,2].set_xlabel("IoU Threshold")
    axs[0,2].set_ylabel("F1 Score")

    axs[1,0].plot(df_mean["IoU_Threshold"], df_mean["True_Positive_Count"])
    axs[1,0].set_title("True Positive Count")
    axs[1,0].set_xlabel("IoU Threshold")
    axs[1,0].set_ylabel("True Positive Count")

    axs[1,1].plot(df_mean["IoU_Threshold"], df_mean["False_Positive_Count"])
    axs[1,1].set_title("False Positive Count")
    axs[1,1].set_xlabel("IoU Threshold")
    axs[1,1].set_ylabel("False Positive Count")

    axs[1,2].plot(df_mean["IoU_Threshold"], df_mean["False_Negative_Count"])
    axs[1,2].set_title("False Negative Count")
    axs[1,2].set_xlabel("IoU Threshold")
    axs[1,2].set_ylabel("False Negative Count")

    plt.savefig("png_files/" + NAME + "_recall_precision_F1_score_TP_FP_FN.png")

