"""Expose the requested split through the field expected by native adapters."""


class NativeSplitView:
    def __init__(self, dataset, split):
        if split not in {'val','test'} or getattr(dataset, 'split', None) != split:
            raise ValueError('Native dataset view must match the selected split')
        self.dataset = dataset
        self.test = getattr(dataset, split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
