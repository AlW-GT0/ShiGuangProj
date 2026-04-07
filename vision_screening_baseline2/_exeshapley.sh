echo "Preprocessing data to fit analysis..."
python ./src/double_data.py
echo "Spliting full dataset into train and test..."
python ./src/split_dataset.py
echo "Training models..."
time python ./src/train_models.py
echo "Applying models for calculating shapley value"
echo "It might take considerable time, please wait..."
time python ./src/calc_importance.py
echo "Finished."

