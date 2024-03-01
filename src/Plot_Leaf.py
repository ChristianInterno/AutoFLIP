import os
import json
import matplotlib.pyplot as plt

# Define the base path
base_path = "/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/result_paper_LEAF/Leaf/"

# Define your datasets, models, algorithms, and the folder name which includes the IDs
datasets = ["Shakespeare", "FEMNIST"]
models = ["NextCharLSTM", "CNNFEMNIST"]
algorithms = ["FedAvg", "FedExp", "FedExpProx", "FedProx"]
folderresults_ids = ["Folderresult1/ID1.json", "Folderresult2/ID2.json"]

# Initialize a dictionary to store the acc1 weighted data for each configuration
acc1_weighted_data = {}

# Loop through each configuration and load the JSON data
for dataset in datasets:
    for model in models:
        for algorithm in algorithms:
            for folderresult_id in folderresults_ids:
                # Construct the file path
                file_path = os.path.join(base_path, dataset, model, algorithm, folderresult_id)

                # Check if the file exists
                if os.path.exists(file_path):
                    # Load the JSON data
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                    # Extract the acc1 weighted metric from clients_updated for each round
                    rounds = sorted(data.keys(), key=int)  # Sort the rounds numerically
                    acc1_weighted = [data[round]['clients_updated']['acc1']['weighted'] for round in rounds]

                    # Store the data in the dictionary
                    config_name = f"{dataset}_{model}_{algorithm}_{folderresult_id.split('/')[0]}"
                    acc1_weighted_data[config_name] = acc1_weighted

# Now plot the data for each configuration
plt.figure(figsize=(15, 7))

for config, acc1_weighted in acc1_weighted_data.items():
    plt.plot(rounds, acc1_weighted, marker='o', linestyle='-', label=config)

plt.title('Trend of Acc1 Weighted in Clients Updated Across Rounds')
plt.xlabel('Round')
plt.ylabel('Acc1 Weighted')
plt.legend()
plt.grid(True)
plt.show()
