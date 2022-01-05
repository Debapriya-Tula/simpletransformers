from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
from litcm import LIT


def utteranceCMI(sentence, lang):
    x = sentence.split()
    langs = [tok.split("/")[-1] for tok in x]
    count = [0, 0, 0]
    listOfUtterance = langs
    count[0] = listOfUtterance.count("en")
    count[1] = listOfUtterance.count(lang)
    count[2] = listOfUtterance.count("O")
    max_cnt = max(count[:2])
    N = count[0] + count[1]
    if N == 0:
        return 0
    P = 0
    prev = listOfUtterance[0]
    for i in range(1, len(listOfUtterance)):
        if listOfUtterance[i] in ["en", lang]:
            if prev != listOfUtterance[i]:
                P += 1
                prev = listOfUtterance[i]
    # Cu = ((N-max_cnt+P)*100.0)/(2.0*N)
    Cu = (N - max_cnt + P) / (2.0 * N)
    return Cu


def calculate_cmi(input_df, lang):
    lang_to_litlang_map = {"Tamil": "tam", "malayalam": "mal", "Kannada": "kan"}
    lang = lang_to_litlang_map[lang]

    lit = LIT(labels=["eng", lang], transliterate=False)

    output_df = pd.DataFrame(columns=["Input", "Label"])
    output_df["Input"] = input_df["Input"].apply(lambda text: lit.identify(text))
    cmi_list = list(map(lambda x: utteranceCMI(x, lang), output_df["Input"]))
    cmi_mean = cmi_list.mean()
    return cmi_mean


class CMILoss(nn.Module):
    def __init__(self, weight=None, alpha=1.7, gamma=0.25):
        super(CMILoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, processed_input_df, base_lang):
        cmi_list = list(
            map(lambda x: utteranceCMI(x, base_lang), processed_input_df["Input"])
        )
        cmi_mean = np.array(cmi_list).mean()

        # LW=α∗CE∗(1−β)γ+α∗CE∗βγ
        # CMI instead of β
        term1 = (
            self.alpha * F.cross_entropy(input, target, self.weight) * (1 - cmi_mean) ** self.gamma
        )
        term2 = self.alpha * F.cross_entropy(input, target, self.weight) * cmi_mean ** self.gamma
        return term1 + term2
