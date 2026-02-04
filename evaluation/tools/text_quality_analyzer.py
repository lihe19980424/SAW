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

# =======================================================
# text_quality_analyzer.py
# Description: Analyze text quality using various metrics
# =======================================================

import math
import torch
import sacrebleu
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import CodeExecutionError, InvalidAnswerError

from bert_score import score


class TextQualityAnalyzer:
    """Base class for text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class DirectTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for direct text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class ReferencedTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for referenced text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference):
        pass


class ExternalDiscriminatorTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for external discriminator text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text1: str, text2: str, description: str):
        pass


class PPLCalculator(DirectTextQualityAnalyzer):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
            Initialize the perplexity calculator.

            Parameters:
                model: The language model for perplexity calculation.
                tokenizer: The tokenizer for the language model.
                device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    # 核心代码
    def analyze(self, text: str):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)
        return ppl.item()


class LogDiversityAnalyzer(DirectTextQualityAnalyzer):
    """Log diversity analyzer for text quality analysis."""
    
    def __init__(self) -> None:
        super().__init__()

    def _eval_text(self, text: str, ngram: int):
        """Evaluate text to compute the number of unique and total n-grams."""
        tokens = text.split()
        ngram_set = set()
        total_ngrams = 0

        for i in range(len(tokens) - ngram + 1):
            ngram_set.add(" ".join(tokens[i:i + ngram]))
            total_ngrams += 1

        return len(ngram_set), total_ngrams

    def _eval_one_instance(self, text: str, ngram_list: list):
        """Evaluate a single text instance for multiple n-gram lengths."""
        results = {}
        for n in ngram_list:
            unique, total = self._eval_text(text, n)
            results[n] = {"unique": unique, "total": total}
        unique_tokens = set(text.split())
        return results, unique_tokens

    def analyze(self, text: str):
        """Analyze text to compute log diversity based on n-gram uniqueness."""
        ngram_list = [2, 3, 4]  # 定义n-gram的长度列表
        # 初始化预测结果字典，包含每个n-gram长度的唯一数和总数
        prediction_results = {n: {"unique": 0, "total": 0} for n in ngram_list}
        unique_token_set = set()  # 初始化唯一token集合

        stripped_text = text.strip()  # 去除文本首尾空白字符
        # 计算n-gram结果和唯一token集合
        ngram_results, unique_tokens = self._eval_one_instance(stripped_text, ngram_list)

        unique_token_set.update(unique_tokens)# 更新唯一token集合

        # 遍历n-gram长度列表，更新预测结果
        for n in ngram_list:
            prediction_results[n]["unique"] += ngram_results[n]["unique"]
            prediction_results[n]["total"] += ngram_results[n]["total"]


        # add
        if prediction_results[n]["total"] == 0:
            # 可以根据具体情况选择合适的处理方式，例如返回特定值或抛出异常
            print("Warning: total is zero, cannot perform division.")
            diversity_scores = [0.000167080745341574, 0.0, 0.0]  # 或者其他合适的默认值
        else:
            # 计算每个n-gram长度的多样性分数
            diversity_scores = [
                1 - (prediction_results[n]["unique"] / prediction_results[n]["total"])
                for n in ngram_list
            ]

        # 计算整体多样性，为各个n-gram长度多样性的乘积
        overall_diversity = (1 - diversity_scores[0] / 100) * (1 - diversity_scores[1] / 100) * (1 - diversity_scores[2] / 100)
        # 计算对数多样性
        log_diversity = -math.log(max(1 - overall_diversity, math.exp(-20)))

        return log_diversity


class BLEUCalculator(ReferencedTextQualityAnalyzer):
    """BLEU calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the BLEU score of the given text with the reference."""
        b = sacrebleu.corpus_bleu([text], [[reference]]).score
        return b


class BERTScoreCalculator(ReferencedTextQualityAnalyzer):
    """BERTScore calculator for text quality analysis."""

    # 修改 __init__ 方法，增加 num_layers 参数
    def __init__(self, lang="en", device='cuda', model_type=None, num_layers=None) -> None:
        self.device = device
        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers # 保存 num_layers

    def analyze(self, text: str, reference: str):
        try:
            # 修改 score 调用，传入 num_layers
            # 当 model_type 是本地路径时，必须传入 num_layers，否则会报错 KeyError
            P, R, F1 = score([text], [reference], lang=self.lang, verbose=False, 
                             device=self.device, model_type=self.model_type, 
                             num_layers=self.num_layers)
            return F1.mean().item()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return 0.0

class PassOrNotJudger(ReferencedTextQualityAnalyzer):
    """Pass or not judger for text quality analysis."""
    def __init__(self) -> None:
        pass

    def _check_correctness(self, prompt: str, completion: str, test: str, entry_point: str):
        """Check the correctness of the code.""" 
        check_program = (
            prompt + '\n' + completion + "\n" +
            test + "\n" +
            f"check({entry_point})"
        )
        # print(check_program)
        try:
            exec_globals = {}
            exec(check_program, exec_globals)
            return 1
        except BaseException as e:
            return 0

    def analyze(self, text: str, reference: dict):
        """Check if the text passes the correctness test."""
        passed = self._check_correctness(reference['task'], text, reference['test'], reference['entry_point'])
        return passed
    

class GPTTextDiscriminator(ExternalDiscriminatorTextQualityAnalyzer):
    """GPT text discriminator for text quality analysis."""

    def __init__(self, openai_model: str, task_description: str) -> None:
        """
            Initialize the GPT text discriminator.

            Parameters:
                openai_model (str): The OpenAI model to use for text discrimination.
                task_description (str): The description of the task for text discrimination.
        """
        self.openai_model = openai_model
        self.task_description = task_description
    
    def _get_query(self, text1: str, text2: str, question: str):
        """Get the query for text discrimination."""

        query = f"Task Description: {self.task_description}\n"
        query += f"Question: {question}\n"
        query += f"Answer 1: {text1}\n"
        query += f"Answer 2: {text2}\n"
        query += f"Which anwser is better? Only return a number."
        query += f"Return 1 if the first text is better, 2 if the second text is better, 0 if they are equal."
        return query

    def analyze(self, text1: str, text2: str, question: str):
        """Analyze the text to determine which one is better."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, 
                                system_content="You are a helpful assistant to determine which of the two answers is better based on the given task description.")
        query = self._get_query(text1, text2, question)
        answer = openai_util.get_result(query)
        # validate answer
        if answer not in ['0', '1', '2']:
            raise InvalidAnswerError
        return eval(answer)