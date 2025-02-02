from python.trainables import TorchTrainable
import torch


class GeneticAdversary:

    def __init__(self, trainable: TorchTrainable, N: int = 10, epochs=500,
                 selective_pressure: float = 0.2, mutation_size=0.001, asexual_repro: float = 1, epsilon: float = 0.01,
                 uncertainty_power: int = 2, sameness_power: int = 4):

        """
        :param x: Batch of NX1X28x28 torch tensor images.
        :param y: Batch of targets NX1. (NO ONE HOT ENCODING)
        :param trainable: Trainable targeted by the attack.
        :param N; Size of the population during the simulation. Strongly recommended to use an even integer.
        :param epochs: Number of epochs to run the genetic algorithm.
        :param selective_pressure: Percentage of the most fit population that are considered during reproduction.
        :param asexual_repro: Percentage of the population that will reproduce asexually (cloning)
        :param epsilon: Coefficient for the linear combination between uncertainty and sameness in the loss function.
        :param uncertainty_power: Power of the exponent used in the uncertainty term of the loss function.
        :param sameness_power: Power of the exponent used in the sameness term in the loss function.
        """
        self.trainable = trainable
        self.images, self.targets = torch.unsqueeze(self.trainable.test_ds.data.to(self.trainable.device), dim=1), self.trainable.test_ds.targets.to(self.trainable.device)

        # Genetic algorithm params
        self.N = N
        self.epochs = epochs
        self.selective_pressure = selective_pressure
        self.mutation_size = mutation_size
        self.asexual_repro = asexual_repro

        # Objective function params
        self.epsilon = epsilon
        self.uncertainty_power = uncertainty_power
        self.sameness_power = sameness_power

        self.history = {
            'certainty': [],
            'distance': [],
            'predictions': [],
            'solutions': [],
        }

    def genetic_attack(self, image, target):

        empty_dim = (self.N,) + self.images.shape[1:]
        population = torch.zeros(empty_dim).to(self.trainable.device)
        curr_solution = None

        for i in range(self.epochs):

            quality = []
            for n in range(self.N):
                quality.append(self._evaluate_quality(population[n], image, target))

            rank = torch.argsort(torch.tensor(quality).to(self.trainable.device), descending=True)
            if i % 1000 == 0:
                print(f'epoch: {i}')
                print(f'loss: {quality[rank[-1]]}')

            parents_idx = []
            for n in range(self.N // 2):
                parents = self.select_parents(rank)
                parents_idx.append(parents)
            parents_idx = torch.stack(parents_idx)

            children = self.generate_children(population, parents_idx)
            children += torch.logical_not(torch.eq(image, 0)) * torch.normal(0, self.mutation_size, size=children.shape).to(self.trainable.device)
            children = torch.clamp(children, -1, 1)

            curr_solution = population[rank[-1]]
            population = children
            population[0] = curr_solution

            if i % 50 == 0:
                self.history['distance'].append(torch.sum(curr_solution ** self.sameness_power).detach().item())
                self.history['certainty'].append(self.trainable(image + curr_solution)[0][target].detach().item())
                self.history['predictions'].append(self.trainable(image + curr_solution).cpu().detach().numpy())
                self.history['solutions'].append(curr_solution.cpu().detach().numpy())

        return curr_solution

    def generate_children(self, population, parents_idx):

        parent_pool = torch.clone(parents_idx)

        # Asexual reproduction (cloning)
        repro_mask = torch.bernoulli(torch.Tensor([self.asexual_repro for n in range(self.N//2)]))
        mask_idx = torch.where(repro_mask == 1)[0]  # Apply mask and find parents to clone
        clones = population[parents_idx[mask_idx]]
        clones = torch.flatten(clones, start_dim=0, end_dim=1)

        # Pool reserved to sexual repro
        parent_pool = parent_pool[self.complement_idx(mask_idx, dim=self.N//2)]

        # Reshape the parents index tensor in preparation for ''fancy'' indexing
        inv_parents_idx = parent_pool.resize(parent_pool.shape[1], parent_pool.shape[0])

        # Sexual reproduction (gene sharing)
        r = torch.rand(size=population[0].shape).to(self.trainable.device)
        batch1 = r * population[inv_parents_idx[0]] + (1 - r) * population[inv_parents_idx[1]]
        batch2 = r * population[inv_parents_idx[1]] + (1 - r) * population[inv_parents_idx[0]]
        children = torch.cat([batch1, batch2])

        # Concatenate clones and new genes
        children = torch.cat([clones, children])

        return children

    def complement_idx(self, idx: torch.Tensor, dim: int):

        """
        Compute the following complement: set(range(dim)) - set(idx).
        SOURCE:  https://stackoverflow.com/questions/67157893/pytorch-indexing-select-complement-of-indices
        :param idx: indexes to be excluded in order to form the complement
        :param dim: max index for the complement
        """
        a = torch.arange(dim, device=idx.device)
        ndim = idx.ndim
        dims = idx.shape
        n_idx = dims[-1]
        dims = dims[:-1] + (-1,)
        for i in range(1, ndim):
            a = a.unsqueeze(0)
        a = a.expand(*dims)
        masked = torch.scatter(a, -1, idx, 0)
        compl, _ = torch.sort(masked, dim=-1, descending=False)
        compl = compl.permute(-1, *tuple(range(ndim - 1)))
        compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
        return compl

    def _evaluate_quality(self, perturbation, image, target):
        distance = torch.sum(perturbation ** self.sameness_power)
        preds = self.trainable(image + perturbation)
        certainty = preds[:, target]
        return certainty + self.epsilon * distance

    def select_parents(self, rank: torch.Tensor):

        """
        :param rank: The descending ranking of the population in terms of quality.
        :return: The index of 2 individuals chosen to become parents.
        """

        # TODO: optimize (naive implementation)

        # We select the first
        lower_bound = int((1 - self.selective_pressure) * self.N)
        first_index = torch.randint(low=lower_bound, high=self.N, size=(1,), device=self.trainable.device)

        # Choose best parent randomly according to selective pressure
        first_parent = rank[first_index]

        # Choose second parent
        second_parent = first_parent
        while second_parent == first_parent:
            second_parent = torch.randint(0, self.N - 1, size=(1,), device=self.trainable.device)

        return torch.tensor([first_parent, second_parent], device=self.trainable.device)