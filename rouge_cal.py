import json
import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util


def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens

def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4
    return bleu

def similariry_score(str1, str2, model):
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score

def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l