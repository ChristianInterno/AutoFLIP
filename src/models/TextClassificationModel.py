from torch import nn


class TextClassificationModel(nn.Module):
    def __init__(self, num_embeddings, embedding_size, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_size, sparse=False)
        self.fc = nn.Linear(embedding_size, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, seq_len):
        embedded = self.embedding(text, seq_len)
        return self.fc(embedded)