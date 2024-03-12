import random
from itertools import combinations, product
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset


class RandomBinaryData:
    def __init__(self, level=4, num_categories=5, num_cousin=2, examples_per_category=2, cousin_size=-1):
        self.output: Optional[torch.Tensor] = None
        self.input = None
        self.level = level
        self.num_categories = num_categories
        self.num_cousins = num_cousin
        self.examples_per_dimension = examples_per_category
        self.cousin_size = cousin_size
        if self.cousin_size == -1:
            self.cousin_size = max(1, num_categories // 5)
        self.all_cousins = list(combinations(range(self.num_categories), self.cousin_size))
        self.subset_cousins = random.sample(self.all_cousins, self.num_cousins)

        if self.cousin_size >= self.num_categories:
            raise ValueError(
                f"The number of categories in a cousin ({self.cousin_size}) must be smaller than the total number of"
                f" categories ({self.num_categories})")

        if self.num_cousins > len(self.all_cousins):
            raise ValueError(
                f"The number of cousins ({self.num_cousins}) cannot be greater than the total number of possible cousin"
                f" combinations ({len(self.all_cousins)}).")

        self._generate_examples()

    @staticmethod
    def _sample_without_repetition(size, maximum):
        if size > maximum:
            raise ValueError(f"Size larger than max. Size: {size}, Max: {maximum}. Possibly too many examples per"
                             f" category required; or insufficient depth")
        lista = []
        cnt = 0
        while cnt < size:
            n = random.randrange(0, maximum)
            if n not in lista:
                lista.append(n)
                cnt += 1
        return lista

    def _generate_examples(self):
        numbers = self._sample_without_repetition(self.num_categories * self.examples_per_dimension, 2 ** self.level)
        data = []
        for num in numbers:
            arr = [int(x) for x in bin(num)[2:]]
            arr = torch.nn.functional.pad(torch.tensor(arr), (self.level - len(arr), 0), 'constant')
            data.append(list(arr.float()))
        output = [[float(c)] for c, _ in product(range(self.num_categories), range(self.examples_per_dimension))]
        self.input, self.output = torch.tensor(data), torch.tensor(output)

    def _permute_values(self, torch_input, torch_output):
        indexes = torch.randperm(torch_input.shape[0])
        torch_output2 = torch_output[indexes]
        torch_input2 = torch_input[indexes]
        return torch_input2, torch_output2

    def data(self, cousin_id=-1):
        if cousin_id == -1:
            torch_input = self.input
            torch_output = self.output
        else:
            if self.num_cousins == 0:
                raise ValueError("No cousins defined")
            if cousin_id >= self.num_cousins:
                raise ValueError("The called cousin does not exist")

            cousins_output = []
            cousins_input = []
            cousin = self.subset_cousins[cousin_id]

            for condition in cousin:
                ii = torch.where(self.output == condition)
                for i in ii[0]:
                    cousins_output.append(self.output[i])
                    cousins_input.append(self.input[i])

            torch_input = torch.stack(cousins_input)
            torch_output = torch.stack(cousins_output)

        torch_input, torch_output = self._permute_values(torch_input, torch_output)
        labels = F.one_hot(torch_output.type(torch.int64), self.num_categories)
        labels = torch.squeeze(labels, 1)
        labels = labels.type(torch.float)

        return TensorDataset(torch_input, labels)