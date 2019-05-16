class BatchGenerator:
    """ Makes torchtext datasets play nice with DataLoaders. Copied from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8"""
    def __init__(self, dl, x_field='text', y_field='label'):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X, y)