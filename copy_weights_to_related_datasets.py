import shutil
import glob
import os


TRAININGS_PATH = "TRAININGS/200_140_trainings"

DEST_PATH = "hist_eq_filtered_variations/200_140/datasets"

PREFIX = "filtered_dataset_"

for train_name in sorted(os.listdir(TRAININGS_PATH)):
    train_num = train_name[-1]

    related_dataset_name = PREFIX + train_num

    model_path = os.path.join(TRAININGS_PATH, train_name, "weights", "best.pt")

    dest_path = os.path.join(DEST_PATH, related_dataset_name)

    shutil.copy(model_path, dest_path)

    print(f"The weights of {train_name} is copied to {dest_path}")





