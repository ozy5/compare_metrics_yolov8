import os
import shutil
import pandas as pd
import glob
import numpy as np
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

fig = plt.subplots(figsize =(20, 8))

#create a dataframe to store the percentage difference of FP and FN values of each experiment compared to not-filtered for each experiment

data = {"Experiment_Name":[],"Percentage_Difference_FP":[],"Percentage_Difference_FN":[]}
barWidth = 0.33
br1 = np.arange(len(paths))
br2 = [x + barWidth for x in br1]

for csv_path_exp_1 in paths:


    exp_1_full_name = os.path.basename(csv_path_exp_1)

    exp_1_name = "_".join(exp_1_full_name.split("_")[0:2])

    print(exp_1_name)



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

    # add row to data
    data["Experiment_Name"].append(exp_1_name)
    data["Percentage_Difference_FP"].append(df_diff_percentage["False_Positive_Count"])
    data["Percentage_Difference_FN"].append(df_diff_percentage["False_Negative_Count"])




    
# plot the FP and FN difference seperately for each experiment as percentage to bar plot. include label, edge color, color. each experiment should be seperated
plt.bar(br1, data["Percentage_Difference_FN"], color ='r', width = barWidth,
        edgecolor ='grey', label ='False Negative')
plt.bar(br2, data["Percentage_Difference_FP"], color ='b', width = barWidth,
        edgecolor ='grey', label ='False Positive')



plt.xlabel("IoU Threshold", fontweight ='bold', fontsize = 15)
plt.ylabel("Percentage Difference", fontweight ='bold', fontsize = 15)
plt.title("Percentage Difference of FP and FN Values of Experiments Compared to not-filtered", fontweight ='bold', fontsize = 20)
plt.xticks([r + barWidth for r in range(len(paths))], data["Experiment_Name"])


plt.legend()


plt.savefig("/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/png_files/all_experiments_percentage_difference_FP_FN.png")






