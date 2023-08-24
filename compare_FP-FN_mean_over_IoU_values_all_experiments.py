import os
import shutil
import pandas as pd
import glob
import matplotlib.pyplot as plt

csv_path_exp_2 = "/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/csv_files/not_filtered_recall_precision_F1_score_TP_FP_FN.csv"

exp_2_name = "not_filtered"

#create dataframe 2, Column order is IoU_Threshold,Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count
df_2 = pd.read_csv(csv_path_exp_2)

wanted_IoU_list = [0.1, 0.2, 0.3, 0.4, 0.5]

# eliminate rows with IoU values not in wanted_IoU_list
df_2 = df_2[df_2["IoU_Threshold"].isin(wanted_IoU_list)]

df_2_mean = df_2.mean()

df_2_mean["True_Positive_Count"] = df_2_mean["True_Positive_Count"] * 10
df_2_mean["False_Positive_Count"] = df_2_mean["False_Positive_Count"] * 10
df_2_mean["False_Negative_Count"] = df_2_mean["False_Negative_Count"] * 10

df_2_mean["Recall"] = df_2_mean["Recall"] * 100
df_2_mean["Precision"] = df_2_mean["Precision"] * 100
df_2_mean["F1_Score"] = df_2_mean["F1_Score"] * 100

df_2_mean = df_2_mean.drop(["IoU_Threshold"])

paths = sorted(glob.glob("/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/csv_files/*_recall_precision_F1_score_TP_FP_FN.csv"))[:-1]




for csv_path_exp_1 in paths:

    csv_path_exp_1 = "/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/csv_files/175_100_recall_precision_F1_score_TP_FP_FN.csv"

    exp_1_full_name = os.path.basename(csv_path_exp_1)

    exp_1_name = "_".join(exp_1_full_name.split("_")[0:2])



    #create dataframe 1, Column order is IoU_Threshold,Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count
    df_1 = pd.read_csv(csv_path_exp_1)


    # eliminate rows with IoU values not in wanted_IoU_list
    df_1 = df_1[df_1["IoU_Threshold"].isin(wanted_IoU_list)]


    # get the mean of every column
    df_1_mean = df_1.mean()


    # multiply TP, FP, FN with 10 to get the exact number of TP, FP, FN
    df_1_mean["True_Positive_Count"] = df_1_mean["True_Positive_Count"] * 10
    df_1_mean["False_Positive_Count"] = df_1_mean["False_Positive_Count"] * 10
    df_1_mean["False_Negative_Count"] = df_1_mean["False_Negative_Count"] * 10



    # multiply Recall, Precision, F1_Score with 100 to get percentage
    df_1_mean["Recall"] = df_1_mean["Recall"] * 100
    df_1_mean["Precision"] = df_1_mean["Precision"] * 100
    df_1_mean["F1_Score"] = df_1_mean["F1_Score"] * 100



    # remove IoU_Threshold column
    df_1_mean = df_1_mean.drop(["IoU_Threshold"])


    # take the difference of two dataframes
    df_diff = df_1_mean - df_2_mean

    # take the percentage difference according to df_2_mean
    df_diff_percentage = df_diff / df_2_mean * 100

    # plot the FP and FN difference as bar plot
    df_diff_percentage[["False_Positive_Count", "False_Negative_Count"]].plot.bar(rot=0, ax=plt.gca(), figsize=(15, 10))

plt.title(f"Percentage Difference of False Positive and False Negative Counts for {exp_1_name} and {exp_2_name}")
plt.xlabel("IoU Threshold")
plt.ylabel("Percentage Difference")
plt.legend(loc="upper right")


plt.show()



