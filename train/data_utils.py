import torch


def individual_collate(batch):
    """
    Custom collation function for collate with new implementation of individual samples in data pipeline
    """

    data = batch

    # Assuming there's at least one instance in the batch
    add_data_keys = data[0].keys()
    collected_data = {k: [] for k in add_data_keys}

    for i in range(len(list(data))):
        for k in add_data_keys:
            collected_data[k].append(data[i][k])

    for k in add_data_keys:
        collected_data[k] = torch.stack(collected_data[k])

    return collected_data
