from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
import nltk
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge as RougeC
from pycocoevalcap.cider.cider import Cider
import jieba

#pycoco
class Caption_Scorer():
    def __init__(self, ref, gt, is_english = True):
        
        if is_english:
            self.ref = ref
            self.gt = gt
        else:
            self.ref = self._tokenize_chinese(ref)
            self.gt = self._tokenize_chinese(gt)

        
        self.scorers = [
            (Bleu(4), ['ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4']),
            # (Meteor(), "METEOR"), # need java version 11.0.16+
            (RougeC(), 'ROUGE-L'),
            (Cider(), 'CIDEr'),
            # (Spice(), "SPICE"), # need java version 11.0.16+
        ]

    def _tokenize_chinese(self, texts_dict):
        """Tokenize Chinese text using jieba"""
        tokenized_dict = {}
        for key, texts in texts_dict.items():
            if isinstance(texts, list):
                # Multiple
                tokenized_texts = []
                for text in texts:
                    # Tokenize and spaces
                    tokens = jieba.cut(text)
                    tokenized_texts.append(' '.join(tokens))
                tokenized_dict[key] = tokenized_texts
            else:
                # Single
                tokens = jieba.cut(texts)
                tokenized_dict[key] = [' '.join(tokens)]
        return tokenized_dict
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:

                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                total_scores[method] = score * 100

        return total_scores

def calculate_bleu_scores(reference_captions, generated_captions):
    reference_list = [[nltk.word_tokenize(ref) for ref in refs] for refs in reference_captions]
    generated_list = [nltk.word_tokenize(gen) for gen in generated_captions]

    bleu1 = corpus_bleu(reference_list, generated_list, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(reference_list, generated_list, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(reference_list, generated_list, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(reference_list, generated_list, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4

def calculate_meteor_scores(reference_captions, generated_captions):
    meteor_scores = [single_meteor_score(nltk.word_tokenize(ref), nltk.word_tokenize(pred)) for ref, pred in zip(reference_captions, generated_captions)]
    meteor_avg  = sum(meteor_scores)/len(meteor_scores)
    return meteor_avg *100

def calculate_rouge_scores(reference_captions, generated_captions):
    rouge = Rouge()
    scores = rouge.get_scores(generated_captions, reference_captions, avg=True, ignore_empty=True)#avg=True)
    
    return scores

def evaluate_captioning(reference_captions, generated_captions):
    #bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(reference_captions, generated_captions)
    metero_score = calculate_meteor_scores(reference_captions, generated_captions)
    rouge_method = calculate_rouge_scores(reference_captions, generated_captions)
    #rouge_method = '' #have some bug so comment out, fix later

    ref, gt = evaluate_captioning2(reference_captions, generated_captions)
    scorer = Caption_Scorer(ref, gt)
    total_score = caption_score_dict = scorer.compute_scores()

    for metric in rouge_method:
        # Loop through each subkey (r, p, f) and multiply by 100
        for key in rouge_method[metric]:
            rouge_method[metric][key] *= 100
            
    # Update the keys
    rouge_method['ROUGE-1'] = rouge_method.pop('rouge-1')
    rouge_method['ROUGE-2'] = rouge_method.pop('rouge-2')
    rouge_method['ROUGE-L'] = rouge_method.pop('rouge-l')

    total_score['CIDEr'] = total_score['CIDEr'] / 10
    return total_score['Bleu'], metero_score, rouge_method, total_score['ROUGE-L'], total_score['CIDEr']

def evaluate_captioning2(reference_captions, generated_captions):
    ref = {}
    gt = {}
    for i, (ref_data, gt_data) in enumerate(zip(reference_captions, generated_captions)):
        ref[str(i)] = [ref_data]
        gt[str(i)] = [gt_data]

    return ref, gt
    