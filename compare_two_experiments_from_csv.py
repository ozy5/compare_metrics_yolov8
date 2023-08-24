import pandas as pd
import matplotlib.pyplot as plt
import os


EXP_NAME_1 = "not_filtered_recall_precision_F1_score_TP_FP_FN.csv"

EXP_NAME_2 = "175_100_recall_precision_F1_score_TP_FP_FN.csv"

real_exp_name_1 = "_".join(EXP_NAME_1.split("_")[0:2])

real_exp_name_2 = "_".join(EXP_NAME_2.split("_")[0:2])

CSV_FOLDER_PATH = "/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/csv_files_0.5conf"

# This code will compare the results of two experiments.
# The result will be a 4x3 plot.
# First row will contain Recall, Precision and F1 Score plots.
# Second row will contain TP, FP and FN plots.
# Third row will contain the difference between the two experiments (difference over the first experiment as a ratio) for Recall, Precision and F1 score.
# Fourth row will contain the difference between the two experiments (difference over the second experiment as a ratio) for TP, FP, FN.


#create 4x3 subplots. First row contains recall, precision and F1 score plots. Second row contains TP, FP and FN plots. Third row contains the difference between the two experiments (difference over the first experiment as a ratio) for each metric.
fig, axs = plt.subplots(4, 3, figsize=(18,25))

#add title to the figure including experiment names
fig.suptitle('Recall, Precision, F1 Score, TP, FP, FN vs IoU Threshold\n' + real_exp_name_1 + " vs " + real_exp_name_2)

#set title for each of 9 subplots
axs[0, 0].set_title("Recall")
axs[0, 1].set_title("Precision")
axs[0, 2].set_title("F1 Score")
axs[1, 0].set_title("True Positive Count")
axs[1, 1].set_title("False Positive Count")
axs[1, 2].set_title("False Negative Count")
axs[2, 0].set_title('Recall Diff(%)')
axs[2, 1].set_title('Precision Diff(%)')
axs[2, 2].set_title('F1 Score Diff(%)')
axs[3, 0].set_title('TP Diff(%)')
axs[3, 1].set_title('FP Diff(%)')
axs[3, 2].set_title('FN Diff(%)')

#set x and y axis labels for each of 9 subplots
axs[0, 0].set(xlabel='IoU Threshold', ylabel='Recall')
axs[0, 1].set(xlabel='IoU Threshold', ylabel='Precision')
axs[0, 2].set(xlabel='IoU Threshold', ylabel='F1 Score')
axs[1, 0].set(xlabel='IoU Threshold', ylabel='TP')
axs[1, 1].set(xlabel='IoU Threshold', ylabel='FP')
axs[1, 2].set(xlabel='IoU Threshold', ylabel='FN')
axs[2, 0].set(xlabel='IoU Threshold', ylabel=f"RECALL\n100 * ({real_exp_name_2} - {real_exp_name_1}) / (1 - {real_exp_name_1}) (%)")
axs[2, 1].set(xlabel='IoU Threshold', ylabel=f"PRECISION\n(100 * {real_exp_name_2} - {real_exp_name_1}) / (1 - {real_exp_name_1}) (%)")
axs[2, 2].set(xlabel='IoU Threshold', ylabel=f"F1 SCORE\n100 * ({real_exp_name_2} - {real_exp_name_1}) / (1 - {real_exp_name_1}) (%)")
axs[3, 0].set(xlabel='IoU Threshold', ylabel=f"TP\n100 * ({real_exp_name_2} - {real_exp_name_1}) / {real_exp_name_1} (%)")
axs[3, 1].set(xlabel='IoU Threshold', ylabel=f"FP\n100 * ({real_exp_name_2} - {real_exp_name_1}) / {real_exp_name_1} (%)")
axs[3, 2].set(xlabel='IoU Threshold', ylabel=f"FN\n100 * ({real_exp_name_2} - {real_exp_name_1}) / {real_exp_name_1} (%)")




# # EXP1 PLOTTING PART

#read csv file as pandas dataframe including column names. Column order is IoU_Threshold,Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count
df_exp1 = pd.read_csv(os.path.join(CSV_FOLDER_PATH, EXP_NAME_1))

#plot Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count VS IoU_Threshold for each experiment
#axs[0, 0] = Recall
#axs[0, 1] = Precision
#axs[0, 2] = F1 Score
#axs[1, 0] = TP
#axs[1, 1] = FP
#axs[1, 2] = FN
axs[0, 0].plot(df_exp1["IoU_Threshold"], df_exp1["Recall"], label=real_exp_name_1)
axs[0, 1].plot(df_exp1["IoU_Threshold"], df_exp1["Precision"], label=real_exp_name_1)
axs[0, 2].plot(df_exp1["IoU_Threshold"], df_exp1["F1_Score"], label=real_exp_name_1)
axs[1, 0].plot(df_exp1["IoU_Threshold"], df_exp1["True_Positive_Count"], label=real_exp_name_1)
axs[1, 1].plot(df_exp1["IoU_Threshold"], df_exp1["False_Positive_Count"], label=real_exp_name_1)
axs[1, 2].plot(df_exp1["IoU_Threshold"], df_exp1["False_Negative_Count"], label=real_exp_name_1)


# # EXP2 PLOTTING PART

#read csv file as pandas dataframe including column names. Column order is IoU_Threshold,Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count
df_exp2 = pd.read_csv(os.path.join(CSV_FOLDER_PATH, EXP_NAME_2))

#plot Recall,Precision,F1_Score,True_Positive_Count,False_Positive_Count,False_Negative_Count VS IoU_Threshold for each experiment
#axs[0, 0] = Recall
#axs[0, 1] = Precision
#axs[0, 2] = F1 Score
#axs[1, 0] = TP
#axs[1, 1] = FP
#axs[1, 2] = FN
axs[0, 0].plot(df_exp2["IoU_Threshold"], df_exp2["Recall"], label=real_exp_name_2)
axs[0, 1].plot(df_exp2["IoU_Threshold"], df_exp2["Precision"], label=real_exp_name_2)
axs[0, 2].plot(df_exp2["IoU_Threshold"], df_exp2["F1_Score"], label=real_exp_name_2)
axs[1, 0].plot(df_exp2["IoU_Threshold"], df_exp2["True_Positive_Count"], label=real_exp_name_2)
axs[1, 1].plot(df_exp2["IoU_Threshold"], df_exp2["False_Positive_Count"], label=real_exp_name_2)
axs[1, 2].plot(df_exp2["IoU_Threshold"], df_exp2["False_Negative_Count"], label=real_exp_name_2)


# # DIFFERENCE PLOTTING PARTS

#calculate difference between the two experiments for each metric
#difference is calculated as (EXP2 - EXP1) / EXP1 * 100
#difference is calculated for each metric
df_difference = pd.DataFrame()
df_difference["IoU_Threshold"] = df_exp1["IoU_Threshold"]
df_difference["Recall_Diff"] = (df_exp2["Recall"] - df_exp1["Recall"]) / (1 - df_exp1["Recall"]) * 100
df_difference["Precision_Diff"] = (df_exp2["Precision"] - df_exp1["Precision"]) / (1 - df_exp1["Precision"]) * 100
df_difference["F1_Score_Diff"] = (df_exp2["F1_Score"] - df_exp1["F1_Score"]) / (1 - df_exp1["F1_Score"]) * 100
df_difference["True_Positive_Count_Diff"] = (df_exp2["True_Positive_Count"] - df_exp1["True_Positive_Count"]) / (df_exp1["True_Positive_Count"]) * 100
df_difference["False_Positive_Count_Diff"] = (df_exp2["False_Positive_Count"] - df_exp1["False_Positive_Count"]) / (df_exp1["False_Positive_Count"]) * 100
df_difference["False_Negative_Count_Diff"] = (df_exp2["False_Negative_Count"] - df_exp1["False_Negative_Count"]) / (df_exp1["False_Negative_Count"]) * 100
# df_difference["True_Positive_Count_Diff"] = (df_exp2["True_Positive_Count"] - df_exp1["True_Positive_Count"])
# df_difference["False_Positive_Count_Diff"] = (df_exp2["False_Positive_Count"] - df_exp1["False_Positive_Count"])
# df_difference["False_Negative_Count_Diff"] = (df_exp2["False_Negative_Count"] - df_exp1["False_Negative_Count"])

#plot difference between the two experiments for each metric
#difference is calculated as (EXP2 - EXP1) / EXP1 * 100
#difference is calculated for each metric
axs[2, 0].plot(df_difference["IoU_Threshold"], df_difference["Recall_Diff"], label=real_exp_name_1 + " vs " + real_exp_name_2)
axs[2, 1].plot(df_difference["IoU_Threshold"], df_difference["Precision_Diff"], label=real_exp_name_1 + " vs " + real_exp_name_2)
axs[2, 2].plot(df_difference["IoU_Threshold"], df_difference["F1_Score_Diff"], label=real_exp_name_1 + " vs " + real_exp_name_2)
axs[3, 0].plot(df_difference["IoU_Threshold"], df_difference["True_Positive_Count_Diff"], label=real_exp_name_1 + " vs " + real_exp_name_2)
axs[3, 1].plot(df_difference["IoU_Threshold"], df_difference["False_Positive_Count_Diff"], label=real_exp_name_1 + " vs " + real_exp_name_2)
axs[3, 2].plot(df_difference["IoU_Threshold"], df_difference["False_Negative_Count_Diff"], label=real_exp_name_1 + " vs " + real_exp_name_2)


#draw horizontal line at y=0 for each of 6 difference plots
axs[2, 0].axhline(y=0, color='r', linestyle='-')
axs[2, 1].axhline(y=0, color='r', linestyle='-')
axs[2, 2].axhline(y=0, color='r', linestyle='-')
axs[3, 0].axhline(y=0, color='r', linestyle='-')
axs[3, 1].axhline(y=0, color='r', linestyle='-')
axs[3, 2].axhline(y=0, color='r', linestyle='-')


#set legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')

#save figure
plt.savefig(os.path.join("/home/umut/Desktop/TEST_EXPERIMENTS/compare_metrics_yolov8/png_files_0.5conf", "COMPARISON__" + real_exp_name_1 + "_vs_" + real_exp_name_2 + ".png"))

#plt.show()


