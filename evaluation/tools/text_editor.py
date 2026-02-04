# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import re
import copy
import nltk
import torch
import random
import numpy as np
from tqdm import tqdm
from nltk import pos_tag
from nltk.corpus import wordnet
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import DiversityValueError
from evaluation.tools.oracle import QualityOracle
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM

# add
import random
import re
import json
from typing import Optional, List

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text

class RandomWalkAttack(TextEditor):
    """
        Remove the watermark using the random walk attack (https://arxiv.org/abs/2311.04378) via black-box access to a quality oracle and a perturbaiton oracle.
        (1) Quality oracle can evaluate whether a candidate output is a high-quality response to a prompt.
        (2) Perturbation oracle can modify an output with a nontrivial probability of maintaining quality, 
            and which induces an efficiently mixing random walk on high-quality outputs.
        
        Examplar Usage: 
        '''
        model_name_or_path="meta-llama/Meta-Llama-3-70B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto') 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        perturbation_oracle = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xl", device_map='auto')
        perturbation_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xl")
        quality_oracle = QualityOracle(tokenizer, model, choice_granularity=5, device=device, check_quality='checker')
        span_length = 6
        attack = RandomWalkAttack(perturbation_tokenizer=perturbation_tokenizer, perturbation_oracle=perturbation_oracle,
                                  quality_oracle=quality_oracle,
                                  max_new_tokens=int(2*span_length), min_length=int(1.5*span_length), 
                                  do_sample=True, top_p=0.95, top_k=None, repetition_penalty=1.5)
        '''
    """

    def __init__(self, perturbation_tokenizer: T5Tokenizer, perturbation_oracle: T5ForConditionalGeneration, quality_oracle: QualityOracle,
                       device='cuda', total_steps=200, span_len=6, target_valid_steps=100, **kwargs):
        """
            Parameters:
            perturbation_tokenizer (T5Tokenizer): The tokenizer for the perturbation oracle.
            perturbation_oracle (T5ForConditionalGeneration): The perturbation oracle.
            quality_oracle (QualityOracle): The quality oracle.
            device (str): The device to use for inference.
            span_len (int): The length of the span to mask in each random walk step.
            total_steps (int): The total number of random walk steps.
            target_valid_steps (int): The target number of valid steps.
        """
        self.perturbation_tokenizer = perturbation_tokenizer
        self.perturbation_oracle = perturbation_oracle.eval()
        self.quality_oracle = quality_oracle
        self.device = device
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)
        
        self.span_len = span_len
        self.total_steps = total_steps
        self.target_valid_steps = target_valid_steps
        if self.quality_oracle.check_quality == 'checker':
            from gramformer import Gramformer
            self.gf = Gramformer(models = 1, use_gpu=True)

    def perturb(self, text: str):
        final_input_text = self.mask_text(text)

        # Tokenize the input
        final_input = self.perturbation_tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        # Generate the edited text
        with torch.inference_mode():
            outputs = self.perturbation_oracle.generate(**final_input, **self.gen_kwargs)
        outputs = self.perturbation_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        infilled_text = outputs[0]
        final_output_text = final_input_text.replace('<extra_id_0>', infilled_text)

        return final_output_text

    def edit(self, text: str, prompt: str, backtrack_patience: int = 100, max_attempts: int = 1000):
        """Edit the text using the T5 model."""

        original_response, n_response = text, text
        n_iter, valid_steps = 0, 0
        patience = 0
        cached_response = copy.deepcopy(n_response)
        # Process the input text in sentence windows
        pbar = tqdm(total=None)
        while n_iter < self.total_steps or valid_steps < self.target_valid_steps:
            candidate_response = self.perturb(n_response)

            candidate_response = self.grammatical_error_correction(candidate_response)
            candidate_response = self.remove_incomplete_sentences(candidate_response)
            
            if self.quality_oracle.maintain_quality(prompt, original_response, candidate_response):
                cached_response = n_response
                n_response = candidate_response
                valid_steps += 1
                if valid_steps % 10 == 0:
                    print(f"Original response: {original_response}")
                print(f"Get a better {valid_steps}-th response at step {n_iter}/{self.total_steps}: {n_response}")
                patience = 0
            else:
                patience += 1
            
            if patience > max_attempts:
                break
            elif patience > backtrack_patience:
                n_response = cached_response
                patience = 0
            
            pbar.update(1)
            n_iter += 1
        pbar.close()

        return n_response

    def grammatical_error_correction(self, text):
        sentences = sent_tokenize(text)
        corrected_sents = []
        for sent in sentences:
            corrected_sent = self.gf.correct(sent, max_candidates=1).pop()
            corrected_sents.append(corrected_sent)
        corrected_text = ' '.join(corrected_sents)
        return corrected_text

    def mask_text(self, text):
        words = text.replace('\n', ' \n').split(' ')
        if len(words) == 1:
            return text + ' <extra_id_0> '
        start = np.random.randint(0, len(words) - self.span_len)
        end = start + self.span_len
        masked_text = ' '.join(words[:start]) + ' <extra_id_0> ' + ' '.join(words[end:])
        return masked_text
    
    def contains_verb(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        return any(tag.startswith('VB') for word, tag in tagged_words)

    def remove_incomplete_sentences(self, text):
        sentences = sent_tokenize(text)
        complete_sentences = []
        for sent in sentences:
            if sent.endswith('.') and not self.contains_verb(sent) and not bool(re.match(r'^\d+\.$', sent)):
                continue
            else:
                complete_sentences.append(sent)
        return ' '.join(complete_sentences)

    def correct_text(self, text):
        """Basic punctuation correction"""
        # Replace multiple spaces with a single space
        corrected_text = re.sub(r'\s+', ' ', text)

        # Correct spaces before commas, periods, colons, semicolons, exclamation marks, and question marks
        corrected_text = re.sub(r'\s+([,.;!?])', r'\1', corrected_text)  # Remove space before punctuation
        corrected_text = re.sub(r'([,.;!?])(?!\s)', r'\1 ', corrected_text)  # Ensure space after punctuation if missing

        # Replace multiple occurrences of punctuation marks with a single instance
        # This part targets specific punctuation marks (you can add more as needed)
        corrected_text = re.sub(r'(\.){2,}', '.', corrected_text)
        corrected_text = re.sub(r'(,){2,}', ',', corrected_text)
        corrected_text = re.sub(r'(!){2,}', '!', corrected_text)
        corrected_text = re.sub(r'(\?){2,}', '?', corrected_text)
        corrected_text = re.sub(r'(:){2,}', ':', corrected_text)
        corrected_text = re.sub(r'(;){2,}', ';', corrected_text)

        return corrected_text

class GPTParaphraser(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, openai_model: str, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, system_content="Your are a helpful assistant to rewrite the text.")
        paraphrased_text = openai_util.get_result(self.prompt + text)
        if paraphrased_text is None:
            return text   # fallback: API 失败就返回原文
        return paraphrased_text
    



class DipperParaphraser(TextEditor):
    """Paraphrase a text using the DIPPER model."""

    def __init__(self, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device='cuda',
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1, **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (T5Tokenizer): The tokenizer for the DIPPER model.
                model (T5ForConditionalGeneration): The DIPPER model.
                device (str): The device to use for inference.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                sent_interval (int): The number of sentences to process at a time.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        # Validate diversity settings
        self._validate_diversity(self.lex_diversity, "Lexical")
        self._validate_diversity(self.order_diversity, "Order")
    
    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise DiversityValueError(type_name)

    def edit(self, text: str, reference: str):
        """Edit the text using the DIPPER model."""

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)
        
        # Preprocess the input text
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        
        # Preprocess the reference text
        prefix = " ".join(reference.replace("\n", " ").split())
        
        output_text = ""
        
        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])
            
            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            
            # Tokenize the input
            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            # final_input = {k: v.cuda() for k, v in final_input.items()}
            final_input = {k: v.to(self.device) for k, v in final_input.items()}
            
            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Update the prefix and output text
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        # nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        # 仅在代码开始时下载一次 wordnet
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            # 使用 tokenizer 自带的 mask_token，适配 [MASK] 或 <mask>
            mask_token = self.tokenizer.mask_token if self.tokenizer.mask_token is not None else '[MASK]'
            masked_sentence = words[:i] + [mask_token] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            
            # add by
            # 查找mask位置前，先检查是否存在mask token
            mask_positions = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            if mask_positions.numel() == 0:
                # 处理没有mask的情况（根据需求选择报错或跳过）
                print("Input text does not contain a mask token.")
                return text  # 直接返回原文本
    
            mask_position = mask_positions.item()  # 现在确保有至少一个元素
            # add end
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1
        
        replaced_text = ' '.join(words)

        return replaced_text

# 核心代码
class TruncatePromptTextEditor(TextEditor):
    """Truncate the prompt from the text."""   """把文本去掉提示部分。"""

    def __init__(self) -> None:
        super().__init__()  # 调用父类的初始化方法，确保正确继承 TextEditor 的特性

    def edit(self, text: str, reference=None):  # text: 要编辑的文本字符串    reference: prompt
        """Truncate the prompt from the text."""
        if reference is not None:
            truncated_text = ' '.join(text.split()[len(reference.split()):])  # 将文本按空格拆分成单词列表，并跳过prompt部分
            return truncated_text
        else:
            return text


class TruncateTaskTextEditor(TextEditor):
    """Truncate the task description from the text, used in code generation."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Truncate the task description from the text."""
        if reference is not None:
            truncated_text = text[len(reference):]
            return truncated_text
        else:
            return text
        

class CodeGenerationTextEditor(TextEditor):
    """Process the code generation output, removing the extra parts."""

    def __init__(self) -> None:
        super().__init__()

    def edit(self, text: str, reference=None):
        """Process the code generation output, removing the extra parts."""
        text = text.lstrip("\n")
        text = text.split("\n\n")[0]
        return text


class BackTranslationTextEditor(TextEditor):
    """Translate text from source language to intermediary language, then back to the source language."""

    def __init__(self,
                 translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                 translate_to_source = Translator(from_lang="zh", to_lang="en").translate) -> None:
        """
        Initialize the back translation editor.

        Parameters:
            translate_to_intermediary (function): The function to translate text to the intermediary language.
            translate_to_source (function): The function to translate text to the source language.
        """
        super().__init__()
        self.translate_to_source = translate_to_source
        self.translate_to_intermediary = translate_to_intermediary

    def edit(self, text: str, reference=None):
        intermediary_text = self.translate_to_intermediary(text)
        edit_result = self.translate_to_source(intermediary_text)
        return edit_result



class MisspellingAttack(TextEditor):
    """基于HELM的拼写错误攻击"""
    def __init__(self, ratio: float = 0.3, misspelling_file: str = '/home/lihe/MarkLLM/evaluation/tools/misspellings.json'):
        super().__init__()
        self.ratio = ratio
        self._load_misspellings(misspelling_file)

    def _load_misspellings(self, file_path):   
        with open(file_path, "r") as f:
            self.correct_to_misspelling = json.load(f)
        self.mispelling_pattern = re.compile(
            r"\b({})\b".format("|".join(self.correct_to_misspelling.keys())), 
            flags=re.IGNORECASE
        )

    def _match_case(self, source_word, target_word):
        """匹配原始单词的大小写格式"""
        if source_word.islower():
            return target_word.lower()
        if source_word.isupper():
            return target_word.upper()
        return target_word.capitalize()

    def edit(self, text: str, reference=None):
        def replace_match(match):
            word = match.group(1)
            if random.random() < self.ratio:
                mispelled = random.choice(self.correct_to_misspelling[word.lower()])
                return self._match_case(word, mispelled)
            return word
        return self.mispelling_pattern.sub(replace_match, text)
    
    
class TypoAttack(TextEditor):
    """基于键盘位置的拼写错误攻击"""
    KEY_APPROX = {
        "q": "was", "w": "qesad", "e": "wsdfr", "r": "edfgt", "t": "rfghy",
        "y": "tghju", "u": "yhjki", "i": "ujklo", "o": "iklp", "p": "ol",
        "a": "qwsz", "s": "weadzx", "d": "erfcxs", "f": "rtgvcd", "g": "tyhbvf",
        "h": "yujnbg", "j": "uikmnh", "k": "iolmj", "l": "opk", "z": "asx",
        "x": "sdcz", "c": "dfvx", "v": "fgbc", "b": "ghnv", "n": "hjmb", "m": "jkn"
    }

    def __init__(self, ratio: float = 0.1):
        super().__init__()
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        perturbed = []
        for char in text:
            if char.lower() in self.KEY_APPROX and random.random() < self.ratio:
                candidates = self.KEY_APPROX[char.lower()]
                new_char = random.choice(candidates)
                perturbed.append(new_char.upper() if char.isupper() else new_char)
            else:
                perturbed.append(char)
        return "".join(perturbed)
    

class ContractionAttack(TextEditor):
    """缩写展开攻击(如将don't变为do not)"""
    CONTRACTION_MAP = {
    "aren't": "are not",
    "can't": "cannot",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not"
    }
    
    # CONTRACTION_MAP = {
    # "ain't": "am not",
    # "aren't": "are not",
    # "can't": "cannot",
    # "can't've": "cannot have",
    # "'cause": "because",
    # "could've": "could have",
    # "couldn't": "could not",
    # "couldn't've": "could not have",
    # "didn't": "did not",
    # "doesn't": "does not",
    # "don't": "do not",
    # "hadn't": "had not",
    # "hadn't've": "had not have",
    # "hasn't": "has not",
    # "haven't": "have not",
    # "he'd": "he would",
    # "he'd've": "he would have",
    # "he'll": "he will",
    # "he'll've": "he will have",
    # "he's": "he is",
    # "how'd": "how did",
    # "how'd'y": "how do you",
    # "how'll": "how will",
    # "how's": "how is",
    # "i'd": "i would",
    # "i'd've": "i would have",
    # "i'll": "i will",
    # "i'll've": "i will have",
    # "i'm": "i am",
    # "i've": "i have",
    # "isn't": "is not",
    # "it'd": "it would",
    # "it'd've": "it would have",
    # "it'll": "it will",
    # "it'll've": "it will have",
    # "it's": "it is",
    # "let's": "let us",
    # "ma'am": "madam",
    # "mayn't": "may not",
    # "might've": "might have",
    # "mightn't": "might not",
    # "mightn't've": "might not have",
    # "must've": "must have",
    # "mustn't": "must not",
    # "mustn't've": "must not have",
    # "needn't": "need not",
    # "needn't've": "need not have",
    # "o'clock": "of the clock",
    # "oughtn't": "ought not",
    # "oughtn't've": "ought not have",
    # "shan't": "shall not",
    # "sha'n't": "shall not",
    # "shan't've": "shall not have",
    # "she'd": "she would",
    # "she'd've": "she would have",
    # "she'll": "she will",
    # "she'll've": "she will have",
    # "she's": "she is",
    # "should've": "should have",
    # "shouldn't": "should not",
    # "shouldn't've": "should not have",
    # "so've": "so have",
    # "so's": "so as",
    # "that'd": "that would",
    # "that'd've": "that would have",
    # "that's": "that is",
    # "there'd": "there would",
    # "there'd've": "there would have",
    # "there's": "there is",
    # "they'd": "they would",
    # "they'd've": "they would have",
    # "they'll": "they will",
    # "they'll've": "they will have",
    # "they're": "they are",
    # "they've": "they have",
    # "to've": "to have",
    # "wasn't": "was not",
    # "we'd": "we would",
    # "we'd've": "we would have",
    # "we'll": "we will",
    # "we'll've": "we will have",
    # "we're": "we are",
    # "we've": "we have",
    # "weren't": "were not",
    # "what'll": "what will",
    # "what'll've": "what will have",
    # "what're": "what are",
    # "what's": "what is",
    # "what've": "what have",
    # "when's": "when is",
    # "when've": "when have",
    # "where'd": "where did",
    # "where's": "where is",
    # "where've": "where have",
    # "who'll": "who will",
    # "who'll've": "who will have",
    # "who's": "who is",
    # "who've": "who have",
    # "why's": "why is",
    # "why've": "why have",
    # "will've": "will have",
    # "won't": "will not",
    # "won't've": "will not have",
    # "would've": "would have",
    # "wouldn't": "would not",
    # "wouldn't've": "would not have",
    # "y'all": "you all",
    # "y'all'd": "you all would",
    # "y'all'd've": "you all would have",
    # "y'all're": "you all are",
    # "y'all've": "you all have",
    # "you'd": "you would",
    # "you'd've": "you would have",
    # "you'll": "you will",
    # "you'll've": "you will have",
    # "you're": "you are",
    # "you've": "you have"
    # }
    PATTERN = re.compile(r"\b({})\b".format("|".join(CONTRACTION_MAP.keys())), flags=re.IGNORECASE)

    def edit(self, text: str, reference=None):
        return self.PATTERN.sub(lambda m: self.CONTRACTION_MAP[m.group().lower()], text)
    
    
    
# ------------------------- 交换攻击适配 -------------------------
class SwapAttack(TextEditor):
    """基于位置交换的攻击"""
    def __init__(self, 
                 word_swap_ratio: float = 0.1, 
                 sentence_swap_ratio: float = 0.1,
                 word_repeat_ratio: float = 0.05,
                 word_delete_ratio: float = 0.05
                 ):
        super().__init__()
        self.word_swap_ratio = word_swap_ratio
        self.sentence_swap_ratio = sentence_swap_ratio
        self.word_repeat_ratio = word_repeat_ratio
        self.word_delete_ratio = word_delete_ratio
        self.sentence_regex = re.compile(r"([^.!?]+[.!?]?)")
        self.word_regex = re.compile(r"\b\w+\b")

    def edit(self, text: str, reference=None):
        # 句子级交换
        sentences = [s.strip() for s in self.sentence_regex.findall(text) if s]
        for i in range(len(sentences)-1):
            if random.random() < self.sentence_swap_ratio:
                sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
        
        # 单词级操作
        words = self.word_regex.findall(" ".join(sentences))
        # 单词交换
        for i in range(len(words)-1):
            if random.random() < self.word_swap_ratio:
                words[i], words[i+1] = words[i+1], words[i]
        # 单词删除
        words = [w for w in words if random.random() > self.word_delete_ratio]
        # 单词重复
        for i in range(len(words)-1, -1, -1):
            if random.random() < self.word_repeat_ratio:
                words.insert(i, words[i])
        
        return " ".join(words).replace("  ", " ")
    
    
# ------------------------- 其他工具类攻击 -------------------------
class LowercaseAttack(TextEditor):
    """全小写转换攻击"""
    def edit(self, text: str, reference=None):
        return text.lower()


class ExpansionAttack(TextEditor):
    """缩略语扩展攻击(如将do not变为don't)"""
    REVERSE_CONTRACTION_MAP = {v:k for k,v in ContractionAttack.CONTRACTION_MAP.items()}
    PATTERN = re.compile(r"\b({})\b".format("|".join(REVERSE_CONTRACTION_MAP.keys())), flags=re.IGNORECASE)

    def edit(self, text: str, reference=None):
        return self.PATTERN.sub(lambda m: self.REVERSE_CONTRACTION_MAP[m.group().lower()], text)