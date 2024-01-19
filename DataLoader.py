import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=None)
data_loader = DataLoader(dataset=dataset,batch_size=10,shuffle=True,num_workers=2)
print(len(dataset))  # = 数据集的样本数
print(len(data_loader))  # = math.ceil(样本数/batch_size) 即向上取整
