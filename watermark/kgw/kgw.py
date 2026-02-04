# ============================================
# kgw.py
# Description: Implementation of KGW algorithm
# ============================================

import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class KGWConfig(BaseConfig):
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']
        self.f_scheme = self.config_dict['f_scheme']
        self.window_scheme = self.config_dict['window_scheme']
        
        self.temperature_inner = self.config_dict['temperature_inner']

        # self.generation_model = self.transformers_config.model
        # self.generation_tokenizer = self.transformers_config.tokenizer
        # self.vocab_size = self.transformers_config.vocab_size
        # self.device = self.transformers_config.device
        # self.gen_kwargs = self.transformers_config.gen_kwargs
    
        print("algorithm_name: KGW", " temperature_inner", self.temperature_inner)
    
    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'KGW'


class KGWUtils:
    """Utility class for KGW algorithm, contains helper functions."""

    def __init__(self, config: KGWConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW utility class.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
        """
        # 保存配置对象
        self.config = config
        # 初始化随机数生成器，并设置种子
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)
        # 生成词汇表大小的随机排列
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        # 定义 f 方案映射
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        # 定义窗口方案映射
        self.window_scheme_map = {"left": self._get_greenlist_ids_left, "self": self._get_greenlist_ids_self}

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        # 根据配置中的 f 方案选择合适的函数并调用
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))
    
    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        # 计算前缀长度内的所有 token 的乘积
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        # 使用乘积结果作为索引，从 prf 中获取值
        return self.prf[time_result % self.config.vocab_size]
    
    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        # 计算前缀长度内的所有 token 的和
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        # 使用和结果作为索引，从 prf 中获取值
        return self.prf[additive_result % self.config.vocab_size]
    
    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        # 直接使用前缀长度前的 token 作为索引，从 prf 中获取值
        return self.prf[input_ids[- self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        # 计算前缀长度内的所有 token 的最小值，并从 prf 中获取值
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        # 根据配置中的窗口方案选择合适的函数并调用
        return self.window_scheme_map[self.config.window_scheme](input_ids)
    
    def _get_greenlist_ids_left(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        # 重新设置随机数生成器的种子
        self.rng.manual_seed((self.config.hash_key * self._f(input_ids)) % self.config.vocab_size)
        # 计算绿色列表的大小
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        # 生成词汇表大小的随机排列
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)  # [50272]
        # 取前 greenlist_size 个 token 作为绿色列表
        greenlist_ids = vocab_permutation[:greenlist_size]  #[25136]
        return greenlist_ids
    
    def _get_greenlist_ids_self(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via selfHash scheme."""
        # 计算绿色列表的大小
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        # 初始化绿色列表
        greenlist_ids = []
        # 获取 f(x) 的值
        f_x = self._f(input_ids)
        # 遍历词汇表中的每个 token
        for k in range(0, self.config.vocab_size):
            # 计算哈希值
            h_k = f_x * int(self.prf[k])
            # 重新设置随机数生成器的种子
            self.rng.manual_seed(h_k % self.config.vocab_size)
            # 生成词汇表大小的随机排列
            vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
            # 取前 greenlist_size 个 token 作为临时绿色列表
            temp_greenlist_ids = vocab_permutation[:greenlist_size]
            # 如果当前 token 在临时绿色列表中，则将其加入绿色列表
            if k in temp_greenlist_ids:
                greenlist_ids.append(k)
        return greenlist_ids
    
    def _compute_z_score(self, observed_count: int , T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        # 计算期望值
        expected_count = self.config.gamma
        # 计算分子部分
        numer = observed_count - expected_count * T 
        # 计算分母部分
        denom = sqrt(T * expected_count * (1 - expected_count))  
        # 计算 z 分数
        z = numer / denom
        return z
    
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        # 计算需要评分的 token 数量
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        # 检查是否有足够的 token 进行评分
        if num_tokens_scored < 1:
            # raise ValueError(
            #     (
            #         f"Must have at least {1} token to score after "
            #         f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
            #     )
            # )
            green_token_flags = [-1 for _ in range(self.config.prefix_length)]
            return -10.0, green_token_flags

        # 初始化绿色 token 计数和标志列表
        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        # 遍历每个 token
        for idx in range(self.config.prefix_length, len(input_ids)):
            # 获取当前 token
            curr_token = input_ids[idx]
            # 获取当前 token 的绿色列表
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            # 检查当前 token 是否在绿色列表中
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)  # 当前 token 是绿色 token
            else:
                green_token_flags.append(0)  # 当前 token 不是绿色 token
        
        # 计算 z 分数
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: KGWConfig, utils: KGWUtils, *args, **kwargs) -> None:
        """
            Initialize the KGW logits processor.

            Parameters:
                config (KGWConfig): Configuration for the KGW algorithm.
                utils (KGWUtils): Utility class for the KGW algorithm.
        """
        # 保存配置对象
        self.config = config
        # 保存工具类对象
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        # 创建一个与 scores 形状相同的全零张量
        green_tokens_mask = torch.zeros_like(scores)
        # 遍历每个批次的 greenlist token IDs
        for b_idx in range(len(greenlist_token_ids)):
            # 将 greenlist token IDs 对应的位置设为 1
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        # 将 green_tokens_mask 转换为布尔张量
        final_mask = green_tokens_mask.bool()  # [1, 50272]
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        # 对 greenlist_mask 中为 True 的位置的 scores 值增加 greenlist_bias
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias #bias=2.0    score的维度是[1, 50272]   scores[greenlist_mask]的维度是[25136]
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        # scores = scores / self.config.temperature_inner
        # 如果输入序列长度小于前缀长度，则直接返回原始 scores
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        # 初始化一个列表，用于存储每个批次的 greenlist token IDs
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])] #1 # 初始化为空列表

        # 遍历每个批次的输入序列
        for b_idx in range(input_ids.shape[0]):
            # 获取当前批次的 greenlist token IDs
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])   # 25136个绿表id
            # 将 greenlist token IDs 存储到 batched_greenlist_ids 中
            batched_greenlist_ids[b_idx] = greenlist_ids   # 25136个绿表id
        
        # 计算 greenlist mask
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids) #[1,50272]

        # 对 greenlist mask 中为 True 的位置的 scores 值增加 greenlist_bias
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta) # mask socre+2.0
        return scores
    

class KGW(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: str | KGWConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        # 初始化算法配置
        # 检查 algorithm_config 是否为字符串类型
        if isinstance(algorithm_config, str):
            # 如果是字符串，将其作为配置文件路径，创建 KGWConfig 实例
            self.config = KGWConfig(algorithm_config, transformers_config)
        # 检查 algorithm_config 是否为 KGWConfig 实例
        elif isinstance(algorithm_config, KGWConfig):
            # 如果是 KGWConfig 实例，直接将其赋值给 self.config
            self.config = algorithm_config
        else:
            # 如果既不是字符串也不是 KGWConfig 实例，抛出类型错误异常
            raise TypeError("algorithm_config must be either a path string or a KGWConfig instance")
        
        # 初始化工具类，用于辅助方法
        self.utils = KGWUtils(self.config)
        # 初始化 logits 处理器，用于生成带水印的文本
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""


        # 配置生成带水印文本的方法
        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,  # 调用生成模型的生成方法
            logits_processor=LogitsProcessorList([self.logits_processor]),   # 设置 logits 处理器列表
            **self.config.gen_kwargs  # 传递生成配置参数
        )
        
        
        # # Encode image         
        # encoded_image = self.config.generation_tokenizer(images=prompt, return_tensors="pt").to(self.config.device)
        # # Generate watermarked text
        # encoded_watermarked_text = generate_with_watermark(encoded_image["pixel_values"], temperature=self.config.temperature_inner)
        # # Decode 
        # watermarked_text = self.config.generation_tokenizer.decode(encoded_watermarked_text[0], skip_special_tokens=True)
        
        
        # Encode prompt # 对输入提示进行编码
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text # 生成带水印的文本
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt,temperature=self.config.temperature_inner, pad_token_id=self.config.generation_tokenizer.eos_token_id)
        # Decode  # 将生成的编码文本解码为字符串
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""
        if len(text) == 0:
            # raise ValueError("Text cannot be empty")
            print("Text is empty")
            z_score = -10.0
            # Determine if the z_score indicates a watermark  # 根据 return_dict 参数决定返回结果的格式
            is_watermarked = z_score > self.config.z_threshold

            # Return results based on the return_dict flag
            if return_dict:
                return {"is_watermarked": is_watermarked, "score": z_score}  # 返回字典格式的结果
            else:
                return (is_watermarked, z_score)  # 返回元组格式的结果

        # Encode the text # 对输入文本进行编码
        encoded_text = self.config.generation_tokenizer(text=text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method  # 计算 z_score，用于检测水印
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark  # 根据 return_dict 参数决定返回结果的格式
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}  # 返回字典格式的结果
        else:
            return (is_watermarked, z_score)  # 返回元组格式的结果
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text  # 对输入文本进行编码
        encoded_text = self.config.generation_tokenizer(text=text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Compute z-score and highlight values # 计算 z-score 和高亮值，用于可视化
        z_score, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens # 解码单个 token
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())  # 解码 token ID 为字符串
            decoded_tokens.append(token)
        # 返回解码后的 token 列表和高亮值列表，用于可视化
        return DataForVisualization(decoded_tokens, highlight_values)