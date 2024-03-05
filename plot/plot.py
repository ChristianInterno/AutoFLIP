import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_server_evaluated_accuracy_diri(data):
    # Filter for 'acc1' and 'server_evaluated'
    acc1_data = data[(data['MetricName'] == 'acc1') & (data['Setting'] == 'server_evaluated')]

    # Format the alpha values and algorithm names
    acc1_data['Alpha'] = acc1_data['Alpha'].str.replace('alpha_', '0.').str.lstrip('0')
    acc1_data['FLAlgorithm'] = acc1_data['FLAlgorithm'].replace({
        'FedExp': 'FedLExp',
        'FedExpProx': 'FedLExProx'
    })

    # Specify a fixed order for the FL algorithms
    algorithm_order = sorted(acc1_data['FLAlgorithm'].unique())

    # Define a consistent color palette for the FL algorithms
    palette = sns.color_palette("tab10", n_colors=len(algorithm_order))

    # Create line plots
    datasets = acc1_data['Dataset'].unique()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6 * len(datasets)))

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        subset_data = acc1_data[acc1_data['Dataset'] == ds]
        sns.lineplot(data=subset_data, x='Alpha', y='MetricValue', hue='FLAlgorithm', hue_order=algorithm_order,
                     marker='o', ci='sd', ax=ax, palette=palette)

        ax.set_title(f'Server Evaluated Accuracy (acc1) with Error Bars for Dataset: {ds}', fontsize=16)
        ax.set_ylabel('Accuracy (acc1)', fontsize=14)
        ax.set_xlabel('Alpha Value', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(title='FL Algorithm', fontsize=12, title_fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_server_evaluated_accuracy_patho(data):
    # Filter for 'acc1' and 'server_evaluated'
    acc1_data = data[(data['MetricName'] == 'acc1') & (data['Setting'] == 'server_evaluated')]

    # Rename the algorithm names
    acc1_data['FLAlgorithm'] = acc1_data['FLAlgorithm'].replace({
        'FedExp': 'FedLExp',
        'FedExpProx': 'FedLExProx'
    })

    # Define a consistent color palette for the FL algorithms
    palette = sns.color_palette("tab10", n_colors=len(acc1_data['FLAlgorithm'].unique()))

    # Create bar plots
    datasets = acc1_data['Dataset'].unique()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6 * len(datasets)))

    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        subset_data = acc1_data[acc1_data['Dataset'] == ds]
        sns.barplot(data=subset_data, x='FLAlgorithm', y='MetricValue', ax=ax, palette=palette, ci='sd')

        ax.set_title(f'Server Evaluated Accuracy (acc1) for Dataset: {ds}', fontsize=16)
        ax.set_ylabel('Accuracy (acc1)', fontsize=14)
        ax.set_xlabel('FL Algorithm', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.tight_layout()
    plt.show()


def main():
    csv_path = input("Enter the path to your CSV file: ")
    data = pd.read_csv(csv_path)

    if 'Alpha' in data.columns:
        plot_server_evaluated_accuracy_diri(data)
    else:
        plot_server_evaluated_accuracy_patho(data)


if __name__ == "__main__":
    main()
