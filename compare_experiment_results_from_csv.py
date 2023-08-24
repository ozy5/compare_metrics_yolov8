import pandas as pd
import matplotlib.pyplot as plt
import os


CSV_FOLDER_PATH = "/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/csv_files_0.5conf"

CSV_NAMES = os.listdir(CSV_FOLDER_PATH)

#create 2x3 subplots. First row contains recall, precision and F1 score plots. Second row contains TP, FP and FN plots.
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Recall, Precision, F1 Score, TP, FP, FN vs IoU Threshold')



#iterate over csv files
for csv_name in CSV_NAMES:
    current_experiment_name = "_".join(csv_name.split("_")[0:2])

    #read csv file as pandas dataframeincluding column names. Column order is IoU_Threshold,Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count
    df = pd.read_csv(os.path.join(CSV_FOLDER_PATH, csv_name))

    #multiply TP, FP, FN with 10
    df["True_Positive_Count"] = df["True_Positive_Count"] * 10
    df["False_Positive_Count"] = df["False_Positive_Count"] * 10
    df["False_Negative_Count"] = df["False_Negative_Count"] * 10

    #multiply Recall, Precision, F1_Score with 100 to get percentage
    df["Recall"] = df["Recall"] * 100
    df["Precision"] = df["Precision"] * 100
    df["F1_Score"] = df["F1_Score"] * 100
    
    #plot Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count VS IoU_Threshold for each experiment
    #axs[0, 0] = Recall
    #axs[0, 1] = Precision
    #axs[0, 2] = F1 Score
    #axs[1, 0] = TP
    #axs[1, 1] = FP
    #axs[1, 2] = FN
    axs[0, 0].plot(df["IoU_Threshold"], df["Recall"], label=current_experiment_name)
    axs[0, 1].plot(df["IoU_Threshold"], df["Precision"], label=current_experiment_name)
    axs[0, 2].plot(df["IoU_Threshold"], df["F1_Score"], label=current_experiment_name)
    axs[1, 0].plot(df["IoU_Threshold"], df["True_Positive_Count"], label=current_experiment_name)
    axs[1, 1].plot(df["IoU_Threshold"], df["False_Positive_Count"], label=current_experiment_name)
    axs[1, 2].plot(df["IoU_Threshold"], df["False_Negative_Count"], label=current_experiment_name)

    #set title for each subplot
    axs[0, 0].set_title("Recall")
    axs[0, 1].set_title("Precision")
    axs[0, 2].set_title("F1 Score")
    axs[1, 0].set_title("True Positive Count")
    axs[1, 1].set_title("False Positive Count")
    axs[1, 2].set_title("False Negative Count")

    #set x and y axis labels
    axs[0, 0].set(xlabel='IoU Threshold', ylabel='Recall')
    axs[0, 1].set(xlabel='IoU Threshold', ylabel='Precision')
    axs[0, 2].set(xlabel='IoU Threshold', ylabel='F1 Score')
    axs[1, 0].set(xlabel='IoU Threshold', ylabel='True Positive Count')
    axs[1, 1].set(xlabel='IoU Threshold', ylabel='False Positive Count')
    axs[1, 2].set(xlabel='IoU Threshold', ylabel='False Negative Count')



#set legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')

#save figure
plt.savefig(os.path.join("/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/png_files_0.5conf", "all_experiments.png"))


    






