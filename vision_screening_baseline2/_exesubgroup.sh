echo "Preprocessing data to fit analysis..."
python ./src/double_data.py
echo "Spliting full dataset into train and test..."
python ./src/split_dataset.py
echo "Training models..."
time python ./src/train_models.py
echo "Applying models for subgroup analysis..."
time python ./src/subgroup_analysis.py
echo "Finished."


