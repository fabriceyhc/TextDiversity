def prepare_splits(dataset_dict, train_val_split = 0.9, val_test_split = 0.5, label_key=""):
    has_train = has_val = has_test = False
    train_id, val_id, test_id = "train", "valid", "test"
    for split_name in dataset_dict.keys():
        if "train" in split_name:
            has_train = True
            train_id = split_name
        elif "val" in split_name:
            has_val = True
            val_id = split_name
        elif "test" in split_name:
            has_test = True
            test_id = split_name
        else:
            dataset_dict.pop(split_name)

    if has_train and has_val and has_test:
        return dataset_dict
    if has_val and not has_test:
        val_test      = dataset_dict[val_id].train_test_split(train_size=val_test_split)
        train_dataset = dataset_dict[train_id]
        val_dataset   = val_test['train']
        test_dataset  = val_test['test']
    if has_test and not has_val:
        train_val     = dataset_dict[train_id].train_test_split(train_size=train_val_split)
        train_dataset = train_val['train']
        val_dataset   = train_val['test']
        test_dataset  = dataset_dict[test_id]
    if not has_val and not has_test:
        train_val     = dataset_dict[train_id].train_test_split(train_size=train_val_split)
        val_test      = train_val['test'].train_test_split(train_size=val_test_split)
        train_dataset = train_val['train']
        val_dataset   = val_test['train']
        test_dataset  = val_test['test']

    dataset_dict[train_id] = train_dataset
    dataset_dict[val_id]   = val_dataset
    dataset_dict[test_id]  = test_dataset

    dataset_dict = rename_splits(dataset_dict)
    dataset_dict = rename_columns(dataset_dict, label_key)

    return dataset_dict

def rename_columns(dataset_dict, label_key=""):
    text_columns = ["sentence"]
    label_key = "" if label_key == "label" else label_key
    for split_name, dataset in dataset_dict.items():
        for column in dataset.features:
            if column in text_columns:
                dataset_dict[split_name] = dataset.rename_column(column, "text")
            if column in label_key:
                dataset_dict[split_name] = dataset.rename_column(column, "label")
    return dataset_dict

def rename_splits(dataset_dict):
    val_names = ["val", "valid"]
    for split_name, dataset in list(dataset_dict.items()):
        if split_name in val_names:
            dataset_dict["validation"] = dataset_dict[split_name]
            dataset_dict.pop(split_name)
    return dataset_dict

class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs
    
    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)
    
    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()