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

# =============================================
# detection.py
# Description: Pipeline for watermark detection
# =============================================

from tqdm import tqdm
from enum import Enum, auto
from watermark.base import BaseWatermark
from evaluation.dataset import BaseDataset
from evaluation.tools.text_editor import TextEditor
from exceptions.exceptions import InvalidTextSourceModeError
# add by lihe
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, BackTranslationTextEditor
from evaluation.tools.text_editor import TruncateTaskTextEditor
from evaluation.tools.text_editor import CodeGenerationTextEditor
from evaluation.tools.text_editor import MisspellingAttack, TypoAttack, ContractionAttack, SwapAttack, LowercaseAttack, ExpansionAttack
from evaluation.tools.text_quality_analyzer import PPLCalculator, LogDiversityAnalyzer, BLEUCalculator, PassOrNotJudger, GPTTextDiscriminator, BERTScoreCalculator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM, T5ForConditionalGeneration, BertForMaskedLM, AutoTokenizer, LlamaTokenizer, T5Tokenizer, BertTokenizer  # 导入transformers库的模型和tokenizer类
import torch
import time
import nltk
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class DetectionPipelineReturnType(Enum):
    """Return type of the watermark detection pipeline."""  
    FULL = auto()
    SCORES = auto()
    IS_WATERMARKED = auto()


class WatermarkDetectionResult:
    """Result of watermark detection."""  # 用于存储水印检测结果的类

    def __init__(self, generated_or_retrieved_text, edited_text, detect_result) -> None:
        """
            Initialize the watermark detection result.

            Parameters:
                generated_or_retrieved_text: The generated or retrieved text.
                edited_text: The edited text.
                detect_result: The detection result.
        """
        """
        初始化水印检测结果。

        参数:
            generated_or_retrieved_text: 生成的或从数据集中获取的原始文本。
            edited_text: 编辑后的文本（经过文本编辑器处理）。
            detect_result: 水印检测的结果（可能包含是否带有水印、得分等信息）。
        """
        self.generated_or_retrieved_text = generated_or_retrieved_text  # 保存生成的或获取的原始文本
        self.edited_text = edited_text  # 保存经过文本编辑器处理后的文本
        self.detect_result = detect_result  # 保存水印检测的结果
        pass  # 无其他操作


class TextQualityComparisonResult:
    """Result of text quality comparison."""

    def __init__(self, watermarked_text: str, unwatermarked_text: str, 
                 watermarked_quality_score: float, unwatermarked_quality_score) -> None:
        """
            Initialize the text quality comparison result.

            Parameters:
                watermarked_text (str): The watermarked text.
                unwatermarked_text (str): The unwatermarked text.
                watermarked_quality_score (float): The quality score of the watermarked text.
                unwatermarked_quality_score (float): The quality score of the unwatermarked text.
        """
        self.watermarked_text = watermarked_text
        self.unwatermarked_text = unwatermarked_text
        self.watermarked_quality_score = watermarked_quality_score
        self.unwatermarked_quality_score = unwatermarked_quality_score
        pass


class WatermarkDetectionPipeline:
    """Pipeline for watermark detection."""  # 基类：用于水印检测的流水线

    def __init__(self, dataset: BaseDataset, text_editor_list: list[TextEditor] = [], unwatermarked_text_editor_list: list[TextEditor] = [], 
                 show_progress: bool = True, return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES, device="cuda", model_ppl: str = "", tokenizer_ppl:str = "", model_Bert: str = "", tokenizer_Bert:str = "", model_dipper: str = "", tokenizer_dipper:str = "") -> None:
        """
            Initialize the watermark detection pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for the pipeline.
                text_editor_list (list[TextEditor]): The list of text editors.
                show_progress (bool): Whether to show progress bar.
                return_type (DetectionPipelineReturnType): The return type of the pipeline.
        """
        """
        初始化水印检测流水线。

        参数:
            dataset (BaseDataset): 用于流水线的数据集。
            text_editor_list (list[TextEditor]): 文本编辑器的列表，用于编辑文本。
            show_progress (bool): 是否显示进度条。
            return_type (DetectionPipelineReturnType): 流水线的返回类型（例如得分或完整检测结果）。
        """
        self.dataset = dataset  # 将数据集保存在实例中
        self.text_editor_list = text_editor_list  # 将文本编辑器列表保存在实例中
        self.unwatermarked_text_editor_list = unwatermarked_text_editor_list
        self.show_progress = show_progress # 是否显示进度条
        self.return_type = return_type # 流水线的返回类型（如得分、是否带水印等）
        self.device = device
        self.model_ppl = model_ppl
        self.tokenizer_ppl = tokenizer_ppl
        self.model_Bert = model_Bert
        self.tokenizer_Bert = tokenizer_Bert
        self.model_dipper = model_dipper
        self.tokenizer_dipper = tokenizer_dipper
        

    def unwatermark_edit_text(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in self.unwatermarked_text_editor_list:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def unwatermark_edit_text_eli5(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        prompt = f"Answer the following question: {prompt} " 
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in self.unwatermarked_text_editor_list:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def unwatermark_edit_text_wmt16_de_en(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in []:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def unwatermark_edit_text_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), CodeGenerationTextEditor()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def unwatermark_edit_text_human_eval_TruncateTask(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
       
    def _edit_text(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in self.text_editor_list:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_eli5(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        prompt = f"Answer the following question: {prompt} "  
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in self.text_editor_list:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_wmt16_de_en(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in []:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), CodeGenerationTextEditor()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_human_eval_TruncateTask(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_1(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), WordDeletion(ratio=0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本

    def _edit_text_word_D_3(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), WordDeletion(ratio=0.3)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_5(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), WordDeletion(ratio=0.5)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_7(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), WordDeletion(ratio=0.7)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_7_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [WordDeletion(ratio=0.7)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_3_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [WordDeletion(ratio=0.3)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_5_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [WordDeletion(ratio=0.5)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_3_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), WordDeletion(ratio=0.3)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_D_5_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), WordDeletion(ratio=0.5)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    

    def _edit_text_word_S_1(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), SynonymSubstitution(ratio=0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本

    def _edit_text_word_S_3(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), SynonymSubstitution(ratio=0.3)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本

    def _edit_text_word_S_5(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), SynonymSubstitution(ratio=0.5)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
      
    def _edit_text_word_S_7(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), SynonymSubstitution(ratio=0.7)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_S_7_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [SynonymSubstitution(ratio=0.7)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_S_3_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [SynonymSubstitution(ratio=0.3)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_S_5_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [SynonymSubstitution(ratio=0.5)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_S_3_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), SynonymSubstitution(ratio=0.3)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_S_5_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), SynonymSubstitution(ratio=0.5)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_word_S_context(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), ContextAwareSynonymSubstitution(ratio=0.7,
                                                                                        tokenizer=self.tokenizer_Bert,
                                                                                        model=self.model_Bert,
                                                                                        device=self.device)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
  
    def _edit_text_word_S_context_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [ContextAwareSynonymSubstitution(ratio=0.7,
                                                            tokenizer=self.tokenizer_Bert,
                                                            model=self.model_Bert,
                                                            device=self.device)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本  
    
    def _edit_text_doc_P_GPT(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), GPTParaphraser(openai_model='moonshot-v1-8k', prompt="Rewrite the following paragraph:\n ")]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本    
    
    
    def _edit_text_doc_P_dipper(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), DipperParaphraser(tokenizer=self.tokenizer_dipper,
                                                                            model=self.model_dipper, device=self.device,
                                                                            lex_diversity=60, order_diversity=0, sent_interval=1, 
                                                                            max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_doc_P_dipper_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [DipperParaphraser(tokenizer=self.tokenizer_dipper,
                                            model=self.model_dipper, device=self.device,
                                            lex_diversity=60, order_diversity=0, sent_interval=1, 
                                            max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_doc_P_dipper_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), DipperParaphraser(tokenizer=self.tokenizer_dipper,
                                                                        model=self.model_dipper, device=self.device,
                                                                        lex_diversity=60, order_diversity=0, sent_interval=1, 
                                                                        max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_misspelling(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), MisspellingAttack(ratio=0.5)]:   # ratio=0.3
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_misspelling_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [MisspellingAttack(ratio=0.5)]:   # ratio=0.3
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
           
    def _edit_text_misspelling_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), MisspellingAttack(ratio=0.5)]:   # ratio=0.3
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    
    def _edit_text_typo(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), TypoAttack(ratio=0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    
    def _edit_text_typo_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TypoAttack(ratio=0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_typo_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), TypoAttack(ratio=0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    
    def _edit_text_contraction(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), ContractionAttack()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_contraction_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [ContractionAttack()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_contraction_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), ContractionAttack()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_swap(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), SwapAttack(word_swap_ratio=0.1, sentence_swap_ratio = 0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_swap_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [SwapAttack(word_swap_ratio=0.1, sentence_swap_ratio = 0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_swap_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), SwapAttack(word_swap_ratio=0.1, sentence_swap_ratio = 0.1)]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_lowercase(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncatePromptTextEditor(), LowercaseAttack()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_lowercase_NoTruncatePrompt(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [LowercaseAttack()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    def _edit_text_lowercase_human_eval(self, text: str, prompt: str = None):
        """Edit text using text editors."""  """使用文本编辑器编辑文本。"""
        # 遍历所有的文本编辑器，依次对文本进行处理
        for text_editor in [TruncateTaskTextEditor(), LowercaseAttack()]:
            text = text_editor.edit(text, prompt)  # 使用文本编辑器对文本进行编辑
        return text  # 返回编辑后的文本
    
    
    def _generate_or_retrieve_text(self, dataset_index: int, watermark: BaseWatermark):
        """Generate or retrieve text from dataset."""
        # pass  # 此方法将在子类中实现，负责生成或获取指定索引的文本
    
        # 从数据集中获取指定索引的提示符
        prompt = self.dataset.get_prompt(dataset_index)
        # 使用水印算法生成带有水印的文本，并返回结果
        return watermark.generate_watermarked_text(prompt)
    
    def unwatermark_generate_or_retrieve_text(self, dataset_index: int, watermark: BaseWatermark):
        """Generate unwatermarked text from the dataset.""" # 生成或从数据集中获取无水印的文本
        # pass  # 此方法将在子类中实现，负责生成或获取指定索引的无水印的文本
        # 从数据集中获取指定索引的提示符
        prompt = self.dataset.get_prompt(dataset_index)
        # 使用水印算法生成无水印的文本，并返回结果 
        return watermark.generate_unwatermarked_text(prompt)

    def _detect_watermark(self, text: str, watermark: BaseWatermark):
        """Detect watermark in text.""" """生成或从数据集中获取文本（待子类实现）。"""
        """检测文本中的水印。"""
        detect_result = watermark.detect_watermark(text, return_dict=True)
        # 调用水印算法检测文本中的水印，返回结果字典
        return detect_result

    def _get_iterable(self):
        """Return an iterable for the dataset."""  """返回数据集的可迭代对象（待子类实现）。"""
        return range(self.dataset.prompt_nums) # 子类应实现此方法，以提供用于遍历数据集的迭代器

    def _get_progress_bar(self, iterable):
        """Return an iterable possibly wrapped with a progress bar."""  """返回带有进度条的可迭代对象（如果启用了显示进度）。"""
        if self.show_progress:
            # 如果启用了进度条，使用`tqdm`包包装迭代器，以显示进度
            return tqdm(iterable, desc="Processing", leave=True)
        return iterable  # 如果不启用进度条，直接返回迭代器

    def evaluate(self, watermark: BaseWatermark, dataset: str, data_lines: int):
        unwatermark_evaluation_result = []
        evaluation_result = []  # 用于存储评估结果
        # evaluation_result_word_D_1 = []
        # evaluation_result_word_D_3 = []
        # evaluation_result_word_D_5 = []
        evaluation_result_word_D_7 = []
        # evaluation_result_word_S_1 = []
        # evaluation_result_word_S_3 = []
        # evaluation_result_word_S_5 = []
        evaluation_result_word_S_7 = []
        evaluation_result_word_S_context = []
        evaluation_result_doc_P_dipper = []
        # evaluation_result_doc_P_GPT = []
        # evaluation_result_misspelling = []
        evaluation_result_typo = []
        # evaluation_result_contraction = []
        # evaluation_result_swap = []
        # evaluation_result_lowercase = []
        
        ppl_evaluation_result = []
        logdiversity_evaluation_result = []
        BLEU_evaluation_result = []
        BERTScore_evaluation_result = []
        GPT_evaluation_result = []
        Pass_evaluation_result = []
        
        ppl_analyzer=PPLCalculator(model=self.model_ppl,
                                tokenizer=self.tokenizer_ppl,
                                device=self.device)
        logdiversity_analyzer = LogDiversityAnalyzer()  
        BLEU_analyzer =  BLEUCalculator() 
        
        my_local_bert_path = "/home/lihe/models/roberta-base"
        # 初始化时传入 model_type 和 num_layers
        # compositional-bert-large-uncased 是 large 模型，通常有 24 层
        # roberta-base 是 base 模型，通常有 12 层
        BERTScore_analyzer = BERTScoreCalculator(device=self.device, 
                                                 model_type=my_local_bert_path, 
                                                 num_layers=12)
        # GPT_analyzer = GPTTextDiscriminator(openai_model='gpt-4', task_description='Translate the following German text to English.') 
        Pass_analyzer = PassOrNotJudger()
        
        bar = self._get_progress_bar(self._get_iterable())  # 获取带进度条的迭代器
        
        execution_time_unwatermarked_200 = []
        execution_time_watermarked_200 = []
        # <--- 新增：初始化检测时间列表
        execution_time_detect_unwatermarked = []
        execution_time_detect_watermarked = []
        
        # 遍历数据集中的每一个索引
        for index in bar:
            # 获取参考文本
            
            prompt = self.dataset.get_prompt(index)
            if dataset =='eli5':
                prompt = f"Answer the following question: {prompt} "  
            elif dataset =='rocstories':
                natural_text = self.dataset.get_natural_text(index)
                prompt = f"Your task is to continue the story beginning with {prompt} Incorporate the words {natural_text} seamlessly to develop a coherent and compelling narrative."  

            reference = self.dataset.get_reference(index) 
            # 生成无水印文本的核心代码
            start_time_unwatermarked = time.time()
            unwatermark_generated_or_retrieved_text = watermark.generate_unwatermarked_text(prompt)
            end_time_unwatermarked = time.time()
            execution_time_unwatermarked = end_time_unwatermarked - start_time_unwatermarked
            execution_time_unwatermarked_200.append(execution_time_unwatermarked)
            # 生成有水印文本的核心代码
            start_time_watermarked = time.time()
            generated_or_retrieved_text = watermark.generate_watermarked_text(prompt)  # self._generate_or_retrieve_text(index, watermark)
            end_time_watermarked = time.time()
            execution_time_watermarked = end_time_watermarked - start_time_watermarked
            execution_time_watermarked_200.append(execution_time_watermarked)
            
            # print("=====================")
            # print(watermark)
            # print("prompt:")
            # print(prompt)
            # print("unwatermarked_text:")
            # print(unwatermark_generated_or_retrieved_text)
            # print("watermarked_text:")
            # print(generated_or_retrieved_text)
            
            # 无攻击时，使用一般文本编辑器编辑文本
            if dataset =='c4' or dataset =='eli5' or dataset =='multinews' or dataset =='rocstories' or dataset =='cnn_daily_mail':
                unwatermark_edited_text = self.unwatermark_edit_text(unwatermark_generated_or_retrieved_text, prompt)
                edited_text = self._edit_text(generated_or_retrieved_text, prompt)
            elif dataset =='wmt16_de_en' or dataset =='flickr30k':
                # unwatermark_edited_text = self.unwatermark_edit_text_wmt16_de_en(unwatermark_generated_or_retrieved_text, prompt)
                # edited_text = self._edit_text_wmt16_de_en(generated_or_retrieved_text, prompt)
                unwatermark_edited_text = unwatermark_generated_or_retrieved_text
                edited_text = generated_or_retrieved_text
            else:
                unwatermark_edited_text = self.unwatermark_edit_text_human_eval(unwatermark_generated_or_retrieved_text, prompt)
                edited_text = self._edit_text_human_eval(generated_or_retrieved_text, prompt)
                unwatermark_edited_text_TruncateTask = self.unwatermark_edit_text_human_eval_TruncateTask(unwatermark_generated_or_retrieved_text, prompt)
                edited_text_TruncateTask = self._edit_text_human_eval_TruncateTask(generated_or_retrieved_text, prompt)
             
            
            
            # 单词删除和替换攻击的文本编辑
            if dataset =='c4' or dataset =='eli5'or dataset =='multinews' or dataset =='rocstories' or dataset =='cnn_daily_mail':
                # edited_text_word_D_1 = self._edit_text_word_D_1(generated_or_retrieved_text, prompt)
                # edited_text_word_D_3 = self._edit_text_word_D_3(generated_or_retrieved_text, prompt)
                # edited_text_word_D_5 = self._edit_text_word_D_5(generated_or_retrieved_text, prompt)
                edited_text_word_D_7 = self._edit_text_word_D_7(generated_or_retrieved_text, prompt)
                
                # edited_text_word_S_1 = self._edit_text_word_S_1(generated_or_retrieved_text, prompt)
                # edited_text_word_S_3 = self._edit_text_word_S_3(generated_or_retrieved_text, prompt)
                # edited_text_word_S_5 = self._edit_text_word_S_5(generated_or_retrieved_text, prompt)
                edited_text_word_S_7 = self._edit_text_word_S_7(generated_or_retrieved_text, prompt)
                
                edited_text_word_S_context = self._edit_text_word_S_context(generated_or_retrieved_text, prompt)
                
                # edited_text_misspelling = self._edit_text_misspelling(generated_or_retrieved_text, prompt)
                edited_text_typo = self._edit_text_typo(generated_or_retrieved_text, prompt)
                # edited_text_contraction = self._edit_text_contraction(generated_or_retrieved_text, prompt)
                # edited_text_swap = self._edit_text_swap(generated_or_retrieved_text, prompt)
                # edited_text_lowercase = self._edit_text_lowercase(generated_or_retrieved_text, prompt)
            elif dataset =='wmt16_de_en' or dataset =='flickr30k':
                # edited_text_word_D_5 = self._edit_text_word_D_5_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                edited_text_word_D_7 = self._edit_text_word_D_7_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                # edited_text_word_S_5 = self._edit_text_word_S_5_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                edited_text_word_S_7 = self._edit_text_word_S_7_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                edited_text_word_S_context = self._edit_text_word_S_context_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                # edited_text_misspelling = self._edit_text_misspelling_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                edited_text_typo = self._edit_text_typo_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                # edited_text_contraction = self._edit_text_contraction_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                # edited_text_swap = self._edit_text_swap_NoTruncatePrompt(generated_or_retrieved_text, prompt)
                # edited_text_lowercase = self._edit_text_lowercase_NoTruncatePrompt(generated_or_retrieved_text, prompt)
            elif dataset =='human_eval':
                edited_text_word_D_5 = self._edit_text_word_D_5_human_eval(generated_or_retrieved_text, prompt)
                edited_text_word_S_5 = self._edit_text_word_S_5_human_eval(generated_or_retrieved_text, prompt)
                # edited_text_misspelling = self._edit_text_misspelling_human_eval(generated_or_retrieved_text, prompt)
                edited_text_typo = self._edit_text_typo_human_eval(generated_or_retrieved_text, prompt)
                # edited_text_contraction = self._edit_text_contraction_human_eval(generated_or_retrieved_text, prompt)
                # edited_text_swap = self._edit_text_swap_human_eval(generated_or_retrieved_text, prompt)
                # edited_text_lowercase = self._edit_text_lowercase_human_eval(generated_or_retrieved_text, prompt)             
            
            if dataset =='c4' or dataset =='eli5' or dataset =='multinews' or dataset =='rocstories' or dataset =='flickr30k' or dataset =='cnn_daily_mail': 
                edited_text_doc_P_dipper = self._edit_text_doc_P_dipper(generated_or_retrieved_text, prompt)
                # edited_text_doc_P_GPT = self._edit_text_doc_P_GPT(generated_or_retrieved_text, prompt)
            elif dataset =='wmt16_de_en':
                edited_text_doc_P_dipper = self._edit_text_doc_P_dipper_NoTruncatePrompt(generated_or_retrieved_text, prompt)
            else:
                edited_text_doc_P_dipper = self._edit_text_doc_P_dipper_human_eval(generated_or_retrieved_text, prompt)
            
            
            # 检测无攻击的水印    
            if dataset =='c4' or dataset =='eli5' or dataset =='multinews' or dataset =='wmt16_de_en' or dataset =='rocstories' or dataset =='flickr30k' or dataset =='cnn_daily_mail': 
                # <--- 修改开始：统计 无水印文本 的检测时间
                start_time = time.time()
                unwatermark_detect_result = self._detect_watermark(unwatermark_edited_text, watermark)
                execution_time_detect_unwatermarked.append(time.time() - start_time)

                # <--- 修改开始：统计 有水印文本 的检测时间
                start_time = time.time()
                detect_result = self._detect_watermark(edited_text, watermark)
                execution_time_detect_watermarked.append(time.time() - start_time)
            elif dataset =='human_eval':
                # 如果跑 human_eval，建议这里也加，逻辑同上
                start_time = time.time()
                unwatermark_detect_result = self._detect_watermark(unwatermark_edited_text_TruncateTask, watermark)
                execution_time_detect_unwatermarked.append(time.time() - start_time)

                start_time = time.time()
                detect_result = self._detect_watermark(edited_text_TruncateTask, watermark)
                execution_time_detect_watermarked.append(time.time() - start_time)
                
            # 检测单词删除和替换攻击时的水印 
            # detect_result_word_D_1 = self._detect_watermark(edited_text_word_D_1, watermark)
            # detect_result_word_D_3 = self._detect_watermark(edited_text_word_D_3, watermark)
            # detect_result_word_D_5 = self._detect_watermark(edited_text_word_D_5, watermark)
            detect_result_word_D_7 = self._detect_watermark(edited_text_word_D_7, watermark)
                        
            # detect_result_word_S_1 = self._detect_watermark(edited_text_word_S_1, watermark)
            # detect_result_word_S_3 = self._detect_watermark(edited_text_word_S_3, watermark)  
            # detect_result_word_S_5 = self._detect_watermark(edited_text_word_S_5, watermark)
            detect_result_word_S_7 = self._detect_watermark(edited_text_word_S_7, watermark)
            
            detect_result_word_S_context = self._detect_watermark(edited_text_word_S_context, watermark)

            # detect_result_misspelling = self._detect_watermark(edited_text_misspelling, watermark)
            detect_result_typo = self._detect_watermark(edited_text_typo, watermark)
            # detect_result_contraction = self._detect_watermark(edited_text_contraction, watermark)
            # detect_result_swap = self._detect_watermark(edited_text_swap, watermark)
            # detect_result_lowercase = self._detect_watermark(edited_text_lowercase, watermark)
            detect_result_doc_P_dipper = self._detect_watermark(edited_text_doc_P_dipper, watermark)
            # detect_result_doc_P_GPT = self._detect_watermark(edited_text_doc_P_GPT, watermark)
        
            # # 将生成的文本、编辑的文本及检测结果封装为WatermarkDetectionResult并添加到评估结果列表中
            # if dataset =='wmt16_de_en' or dataset =='flickr30k':
            #     unwatermark_evaluation_result.append(WatermarkDetectionResult(unwatermark_generated_or_retrieved_text, unwatermark_generated_or_retrieved_text, unwatermark_detect_result))
            #     evaluation_result.append(WatermarkDetectionResult(generated_or_retrieved_text, generated_or_retrieved_text, detect_result)) 
            
            unwatermark_evaluation_result.append(WatermarkDetectionResult(unwatermark_generated_or_retrieved_text, unwatermark_edited_text, unwatermark_detect_result))
            evaluation_result.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text, detect_result))  

            # evaluation_result_word_D_1.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_D_1, detect_result_word_D_1))
            # evaluation_result_word_D_3.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_D_3, detect_result_word_D_3))
            # evaluation_result_word_D_5.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_D_5, detect_result_word_D_5))
            evaluation_result_word_D_7.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_D_7, detect_result_word_D_7))
                
            # evaluation_result_word_S_1.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_S_1, detect_result_word_S_1))
            # evaluation_result_word_S_3.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_S_3, detect_result_word_S_3))
            # evaluation_result_word_S_5.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_S_5, detect_result_word_S_5))
            evaluation_result_word_S_7.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_S_7, detect_result_word_S_7))
            
            evaluation_result_word_S_context.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_word_S_context, detect_result_word_S_context))
        
            # evaluation_result_misspelling.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_misspelling, detect_result_misspelling))
            evaluation_result_typo.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_typo, detect_result_typo))
            # evaluation_result_contraction.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_contraction, detect_result_contraction))
            # evaluation_result_swap.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_swap, detect_result_swap))
            # evaluation_result_lowercase.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_lowercase, detect_result_lowercase))
            
            evaluation_result_doc_P_dipper.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_doc_P_dipper, detect_result_doc_P_dipper))
            # evaluation_result_doc_P_GPT.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text_doc_P_GPT, detect_result_doc_P_GPT))
            
            # 分析编辑后的文本的质量
            if dataset =='c4' or dataset =='eli5' or dataset =='multinews' or dataset =='rocstories' or dataset == 'flickr30k' or dataset =='cnn_daily_mail':
                ppl_watermarked_quality_score = ppl_analyzer.analyze(edited_text)
                ppl_unwatermarked_quality_score = ppl_analyzer.analyze(unwatermark_edited_text)
                logdiversity_watermarked_quality_score = logdiversity_analyzer.analyze(edited_text)
                logdiversity_unwatermarked_quality_score = logdiversity_analyzer.analyze(unwatermark_edited_text) 
                # <--- 新增：计算 BERTScore 
                # 我们计算 水印文本 相对于 无水印文本（作为参考） 的语义相似度
                BERTScore_watermarked_quality_score = BERTScore_analyzer.analyze(edited_text, unwatermark_edited_text)
                # 无水印对无水印本身是 1.0，但为了保持格式一致也计算一下
                BERTScore_unwatermarked_quality_score = 1.0
                
                
                # 将生成的文本、编辑的文本及分析结果封装为TextQualityComparisonResult并添加到评估结果列表中
                ppl_evaluation_result.append(TextQualityComparisonResult(edited_text, unwatermark_edited_text, 
                                                                        ppl_watermarked_quality_score, ppl_unwatermarked_quality_score))
                logdiversity_evaluation_result.append(TextQualityComparisonResult(edited_text, unwatermark_edited_text, 
                                                                                logdiversity_watermarked_quality_score, logdiversity_unwatermarked_quality_score))
                # BLEU_watermarked_quality_score = BLEU_analyzer.analyze(generated_or_retrieved_text, reference)
                # BLEU_unwatermarked_quality_score = BLEU_analyzer.analyze(unwatermark_generated_or_retrieved_text, reference)
                # BLEU_evaluation_result.append(TextQualityComparisonResult(generated_or_retrieved_text, unwatermark_generated_or_retrieved_text, 
                #                                         BLEU_watermarked_quality_score, BLEU_unwatermarked_quality_score)) 
                # <--- 新增：添加 BERTScore 结果到列表
                BERTScore_evaluation_result.append(TextQualityComparisonResult(edited_text, unwatermark_edited_text, 
                                                                                BERTScore_watermarked_quality_score, BERTScore_unwatermarked_quality_score))
            elif dataset =='wmt16_de_en':
                BLEU_watermarked_quality_score = BLEU_analyzer.analyze(generated_or_retrieved_text, unwatermark_generated_or_retrieved_text)  # reference
                BLEU_unwatermarked_quality_score = BLEU_analyzer.analyze(unwatermark_generated_or_retrieved_text, unwatermark_generated_or_retrieved_text)   # reference
                #GPT_quality_score = GPT_analyzer.analyze(edited_text, unwatermark_edited_text, prompt)
                BLEU_evaluation_result.append(TextQualityComparisonResult(generated_or_retrieved_text, unwatermark_generated_or_retrieved_text, 
                                                                        BLEU_watermarked_quality_score, BLEU_unwatermarked_quality_score))
                #GPT_evaluation_result.append(TextQualityComparisonResult(edited_text, unwatermark_edited_text, 
                #                                                        GPT_quality_score, GPT_quality_score))
            else:
                Pass_watermarked_quality_score = Pass_analyzer.analyze(edited_text, reference)
                Pass_unwatermarked_quality_score = Pass_analyzer.analyze(unwatermark_edited_text, reference)
                Pass_evaluation_result.append(TextQualityComparisonResult(generated_or_retrieved_text, unwatermark_generated_or_retrieved_text, 
                                                                        Pass_watermarked_quality_score, Pass_unwatermarked_quality_score)) 
                        
        execution_time_unwatermarked_200_sum = sum(execution_time_unwatermarked_200)
        execution_time_watermarked_200_sum = sum(execution_time_watermarked_200)
        if len(execution_time_unwatermarked_200) == 0:
            execution_time_unwatermarked_200_sum = 0.0
            execution_time_unwatermarked_200_avg = 0.0
        else:
            execution_time_unwatermarked_200_avg = execution_time_unwatermarked_200_sum/len(execution_time_unwatermarked_200)       
        if len(execution_time_watermarked_200) == 0:
            execution_time_watermarked_200_sum = 0.0
            execution_time_watermarked_200_avg = 0.0
        else:    
            execution_time_watermarked_200_avg = execution_time_watermarked_200_sum/len(execution_time_watermarked_200) 
        
        # <--- 新增：计算检测时间统计
        execution_time_detect_unwatermarked_sum = sum(execution_time_detect_unwatermarked)
        execution_time_detect_watermarked_sum = sum(execution_time_detect_watermarked)

        if len(execution_time_detect_unwatermarked) == 0:
            execution_time_detect_unwatermarked_avg = 0.0
        else:
            execution_time_detect_unwatermarked_avg = execution_time_detect_unwatermarked_sum / len(execution_time_detect_unwatermarked)

        if len(execution_time_detect_watermarked) == 0:
            execution_time_detect_watermarked_avg = 0.0
        else:
            execution_time_detect_watermarked_avg = execution_time_detect_watermarked_sum / len(execution_time_detect_watermarked)
        # <--- 新增结束
        
        # 根据return_type选择返回不同的评估结果
        if self.return_type == DetectionPipelineReturnType.FULL:
            return unwatermark_evaluation_result, evaluation_result, evaluation_result_word_D_7, evaluation_result_word_S_7, evaluation_result_word_S_context, evaluation_result_typo, evaluation_result_doc_P_dipper, ppl_evaluation_result, logdiversity_evaluation_result, BLEU_evaluation_result, BERTScore_evaluation_result, GPT_evaluation_result, Pass_evaluation_result, execution_time_unwatermarked_200_sum, execution_time_watermarked_200_sum, execution_time_unwatermarked_200_avg, execution_time_watermarked_200_avg, execution_time_detect_unwatermarked_sum, execution_time_detect_watermarked_sum, execution_time_detect_unwatermarked_avg, execution_time_detect_watermarked_avg  # 返回完整的评估结果
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_result]  # 仅返回每个结果的得分
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [result.detect_result['is_watermarked'] for result in evaluation_result]  # 返回每个结果是否带有水印

        
class WatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting watermarked text."""
    # 用于检测带有水印文本的流水线，继承自WatermarkDetectionPipeline
    # 初始化检测流水线，传入数据集、文本编辑器列表、是否显示进度、返回类型等参数
    def __init__(self, dataset, text_editor_list=[],
                 unwatermarked_text_editor_list=[], 
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES, *args, **kwargs) -> None:
        # 调用父类的初始化方法，继承父类的参数和功能
        super().__init__(dataset, text_editor_list, unwatermarked_text_editor_list, show_progress, return_type)

    def _get_iterable(self):
        """Return an iterable for the prompts.""" # 返回用于迭代的对象，通常是提示符的数量范围
        # 返回从0到数据集中提示符数量的可迭代范围
        return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Generate watermarked text from the dataset.""" # 生成或从数据集中获取带有水印的文本
        # 从数据集中获取指定索引的提示符
        prompt = self.dataset.get_prompt(dataset_index)
        # 使用水印算法生成带有水印的文本，并返回结果
        return watermark.generate_watermarked_text(prompt)
    
    def unwatermark_generate_or_retrieve_text(self, dataset_index, watermark):
        """Generate watermarked text from the dataset.""" # 生成或从数据集中获取带有水印的文本
        # 从数据集中获取指定索引的提示符
        prompt = self.dataset.get_prompt(dataset_index)
        # 使用水印算法生成无水印的文本，并返回结果 
        return watermark.generate_unwatermarked_text(prompt)
    