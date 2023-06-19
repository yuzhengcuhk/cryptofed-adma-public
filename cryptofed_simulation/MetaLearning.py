#   install learn2learn
import learn2learn as lel
import torchvision
from caffe2.python.core import Net
from learn2learn.data import transforms
from tensorflow_core.contrib.eager.python.examples.l2hmc.l2hmc import compute_loss
from torch import optim
from torchvision.datasets import mnist
import metafunc

data_meta = torchvision.datasets.MMIST(root="../data_meta/mnist", train=True)
data_meta = lel.data.MetaDataset(data_meta)
train_tasks = lel.data.TaskDataset(data_meta,
                                   task_transforms=[
                                       transforms.NWays(mnist, n=3),
                                       transforms.KShots(mnist, k=1),
                                       transforms.LoadData(mnist)],
                                   num_tasks=10)
model_meta = Net()
tmp_maml = lel.algorithms.MAML(model_meta, lr=1e-3, first_order=False)
tmp_opt = optim.Adam(tmp_maml.parameters(), lr=4e-3)  # model after optimization

for iteration in range(metafunc.num_iterations):
    learner_meta = tmp_maml.clone()  # Create a clone of a model
    for task in train_tasks:  # Split task in adaption_task and evaluation_task
        for step in range(metafunc.adaptation_steps):  # Fast adapt
            error_mata = compute_loss(metafunc.adaptation_task)
            learner_meta.adapt(error_mata)

        # Compute evaluation loss
        evaluation_error = compute_loss(metafunc.evaluation_task)

        # Meta-update the model parameters
        tmp_opt.zero_grad()
        evaluation_error.backward()
        tmp_opt.step()