from model import predict_labels

test_file = "./TestFileCreator/test_file.csv"
model_path = "Model"
labels_file = "./Model/label_mapping.json"
output_file = "results.csv"

predict_labels(test_file, model_path, labels_file, output_file)
