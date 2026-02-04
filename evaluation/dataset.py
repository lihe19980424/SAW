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

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json
# add
from datasets import load_dataset
from PIL import Image
import requests

import os
import pandas as pd
import datasets

from transformers import BlipProcessor, BlipForConditionalGeneration


class BaseDataset:
    """Base class for dataset."""

    def __init__(self):
        """Initialize the dataset."""
        self.prompts = []
        self.natural_texts = []
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, data_lines: int):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file.""" 
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[0:self.data_lines]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])
            self.references.append(item['natural_text'])
            


class ELI5Dataset(BaseDataset):
    """Dataset class for ELI5 dataset."""

    def __init__(self, data_source: str, data_lines: int):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[0:self.data_lines]:
            item = json.loads(line)
            self.prompts.append(item['title'][len('ELI5: '):])
            # self.prompts.append(item['title'])
            self.natural_texts.append(item["C1"]["comment_text"][0])
            self.references.append(item["C1"]["comment_text"][0])
            
        
class MULTINEWSDataset(BaseDataset):
    """Dataset class for multinews dataset."""

    def __init__(self, data_source: str, data_lines: int):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[0:self.data_lines]:
            item = json.loads(line)
            self.prompts.append(item['document'])
            self.natural_texts.append(item['summary'])
            self.references.append(item['summary'])
            

class ROCSTORIESDataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, data_lines: int):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[0:self.data_lines]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['constraint_words'])
            self.references.append(item['continuation'])


class CNNDAILYMAILDataset(BaseDataset):
    """Dataset class for cnn_daily_mail dataset."""

    def __init__(self, data_source: str, data_lines: int):
        """
            Initialize the cnn_daily_mail dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the cnn_daily_mail dataset file."""
        with open(self.data_source, 'r') as f:
            data = json.load(f)
            for item in data[0:self.data_lines]:
                self.prompts.append(item.get("output", ""))
                self.natural_texts.append(item.get("input", ""))
                self.references.append(item.get("input", ""))
            

class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(self, data_source: str, data_lines: int) -> None:
        """
            Initialize the WMT16 DE-EN dataset.

            Parameters:
                data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[0:self.data_lines]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(self, data_source: str, data_lines: int) -> None:
        """
            Initialize the HumanEval dataset.

            Parameters:
                data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[0:self.data_lines]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})


class Flickr30kDataset(BaseDataset):
    def __init__(self, data_source: str, data_lines: int):
        super().__init__()
        self.data_source = data_source
        self.data_lines = data_lines
        self.load_data()
        
    def load_data(self):
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, "dataset", "flickr30k", "flickr_annotations_30k.csv")
        image_dir = os.path.join(current_dir, "dataset", "flickr30k", "flickr30k-images")

        df = pd.read_csv(csv_path)
        _JSON_KEYS = ['raw','sentids']
        for c in _JSON_KEYS:
            df[c] = df[c].apply(json.loads)
        # df.head(self.data_lines)    df[0:self.data_lines].iterrows()
        for r_idx, r in df.head(self.data_lines).iterrows():
            r_dict = r.to_dict()
            image_path = os.path.join(image_dir, r_dict['filename'])

            if not os.path.exists(image_path):
                print(f"警告: 文件 {image_path} 不存在，跳过该记录。")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                self.prompts.append(image)
                self.natural_texts.append(r_dict['raw'][0])
                self.references.append(r_dict['raw'][0])
            except Exception as e:
                print(f"处理文件 {image_path} 时发生错误: {e}")

    # def load_data(self):
    #     dataset = load_dataset(self.data_source)
    #     for i in range(min(self.data_lines, len(dataset['train']))):
    #         image = dataset['train'][i]['image']
    #         caption = dataset['train'][i]['captions'][0]
    #         self.prompts.append(image)
    #         self.natural_texts.append(caption)
    #         self.references.append(caption)

if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json')
    d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl')
    d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl')
