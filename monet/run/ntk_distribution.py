import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from monet.ntk.compute_score import compute_score
from monet.utils.CIFAR import CIFAR10Dataset
from monet.utils.helpers import subset_classes
from naslib.predictors import ZeroCost
from naslib.predictors.utils.pruners.measures.epe_nas import compute_epe_score
from naslib.search_spaces import NasBench101SearchSpace
from naslib.search_spaces.nasbench101.conversions import convert_spec_to_model
from naslib.utils import get_dataset_api

SEARCH_SPACE = "nasbench101"
DATASET = "cifar10"
nb101_api = get_dataset_api("nasbench101", 'cifar10')
nb101_data = nb101_api["nb101_data"]

def myfun(arch_hash):
    # print(arch_hash)
    model = NasBench101SearchSpace()
    model.set_spec(arch_hash, nb101_api)
    network = convert_spec_to_model(model.spec).to("cuda")
    n_param = sum([np.prod(p.size()) for p in network.parameters()])

    dataset = CIFAR10Dataset()
    lambda_min, lambda_max, ntk, _ = compute_score(network.model, dataset)
    x = torch.tensor(dataset.__getitem__(0)[0].unsqueeze(0)).to("cuda")
    y = torch.tensor(dataset.__getitem__(0)[1]).to("cuda")

    data_loader = datasets.CIFAR10(root='../../naslib/data', train=False, download=False,
    transform=transforms.Compose([transforms.ToTensor()]))
    zc = ZeroCost("epe_nas")
    epe_score = zc.query(model, data_loader)
    print(f"Epe score for {arch_hash}: {epe_score}")
    return {"lambda_min": lambda_min, "lambda_max": lambda_max, "ntk": ntk, "n_param": n_param, "epe": epe_score}

if __name__ == '__main__':
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    df = pd.read_csv("../csv/nasbench101.csv")
    hash_iterator = nb101_data.hash_iterator()
    for i, hash in tqdm(enumerate(hash_iterator), total=len(hash_iterator)):
        res = myfun(arch_hash=hash)
        lambda_min = res["lambda_min"]
        lambda_max = res["lambda_max"]
        ntk = res["ntk"]
        n_param = res["n_param"]
        epe = res["epe"]
        print(f"Score: {lambda_min}, Lmax: {lambda_max}, Params: {n_param}, Accuracy: {df.loc[df['arch_hash'] == hash, 'cifar_10_val_accuracy'].values[0]}")
        df.loc[df["arch_hash"] == hash, "lambda_min"] = lambda_min
        df.loc[df["arch_hash"] == hash, "lambda_max"] = lambda_max
        # print(ntk.cpu().detach().numpy())
        df.loc[df["arch_hash"] == hash, "epe"] = epe
        df.loc[df["arch_hash"] == hash, "trace"] = torch.trace(ntk).cpu().detach().numpy()
        df.to_csv("ntk_nasbench101.csv")

        if i > 20000:
            break
    sns.scatterplot(data=df, y="cifar_10_val_accuracy", x="score")
    plt.show()
