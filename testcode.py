import torch
print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
