from .lensless_data import lensless_dataset, OperatorDataset

dataset_dict = {
    'lensless': lensless_dataset,
    'operator': OperatorDataset
}