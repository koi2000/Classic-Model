import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim

class Config(object):
    use_gpu = True
    device = "cpu"
    data = "./all.txt"
    dataPath = "./data"