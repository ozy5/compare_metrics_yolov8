import shutil
import glob
import os


TRAININGS_PATH = "TRAININGS"

DATASETS_PATH = "hist_eq_filtered_variations"


PREFIX = "filtered_dataset_"

for train_name in sorted(os.listdir(TRAININGS_PATH)):

    train_actual_name = "_".join(train_name.split("_")[0:2])
        
    DEST_PATH = os.path.join(DATASETS_PATH, train_actual_name, "datasets")

    for sub_train_name in sorted(os.listdir(os.path.join(TRAININGS_PATH, train_name))):

        train_num = sub_train_name[-1]



        if train_actual_name != "not_filtered":
            related_dataset_name = PREFIX + train_num
        else:
            related_dataset_name = "not_filtered_dataset_" + train_num

        model_path = os.path.join(TRAININGS_PATH, train_name, sub_train_name, "weights", "best.pt")

        dest_path = os.path.join(DEST_PATH, related_dataset_name)

        shutil.copy(model_path, dest_path)

        print(f"The weights of {train_name} is copied to {dest_path}")





