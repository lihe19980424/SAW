# ============================================
# kgw.py
# Description: Implementation of KGW algorithm
# ============================================

import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization
# add by lihe 
from scipy.stats import norm
# 导入random模块以设置随机种子，确保结果可复现5
import random

class KGWConfig:
    """Config class for KGW algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the KGW configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/KGW.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'KGW':
            raise AlgorithmNameMismatchError('KGW', config_dict['algorithm_name'])

        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        self.f_scheme = config_dict['f_scheme']
        self.window_scheme = config_dict['window_scheme']
        
        self.p = config_dict['p']
        self.prev_n = config_dict['prev_n']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        

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
    
    # 原来的代码
    # def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
    #     """Score the input_ids and return z_score and green_token_flags."""
    #     # 计算需要评分的 token 数量
    #     num_tokens_scored = len(input_ids) - self.config.prefix_length
    #     # 检查是否有足够的 token 进行评分
    #     if num_tokens_scored < 1:
    #         raise ValueError(
    #             (
    #                 f"Must have at least {1} token to score after "
    #                 f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
    #             )
    #         )
    #     # 初始化绿色 token 计数和标志列表
    #     green_token_count = 0
    #     green_token_flags = [-1 for _ in range(self.config.prefix_length)]
    #     # 遍历每个 token
    #     for idx in range(self.config.prefix_length, len(input_ids)):
    #         # 获取当前 token
    #         curr_token = input_ids[idx]
    #         # 获取当前 token 的绿色列表
    #         greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
    #         # 检查当前 token 是否在绿色列表中
    #         if curr_token in greenlist_ids:
    #             green_token_count += 1
    #             green_token_flags.append(1)  # 当前 token 是绿色 token
    #         else:
    #             green_token_flags.append(0)  # 当前 token 不是绿色 token
        
    #     # 计算 z 分数
    #     z_score = self._compute_z_score(green_token_count, num_tokens_scored)
    #     return z_score, green_token_flags    
     
    def score_sequence(self, input_ids: torch.Tensor, prompt_cut) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        # 计算需要评分的 token 数量   num_tokens_scored=[200]
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        # 检查是否有足够的 token 进行评分
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )
        # 初始化绿色 token 计数和标志列表
        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
        unbiased_watermark_token_count = 0
        kgw_watermark_token_count = 0
        
        # add by lihe
        # 初始化一个集合，用于存储已处理过的上下文代码
        cc_history = set()
        # 忽略历史
        ignore_history = False
        # 初始化空列表，用于放置一条文本的200个token
        diff_200 = []
        # 遍历每个 token #这个循环的作用通常是在跳过前缀部分之后，遍历剩余的令牌（tokens），从而对其进行处理或操作self.config.prefix_length
        for idx in range(0, len(input_ids)-self.config.prev_n):
            # 获取当前 token
            curr_token = input_ids[idx+self.config.prev_n-1]  # curr_token = tensor(5, device='cuda:0')
            
            # 得到复现的watermark_code 开始的位置
            # 导入hashlib库，用于生成哈希值以生成种子
            import hashlib
            # 创建一个SHA-256哈希对象
            m = hashlib.sha256()
            # 根据当前token生成8个字节的二进制数据，每个字节8位，共64位  # input_ids[idx]=tensor(5, device='cuda:0') # context_code = b'\x05\x00\x00\x00\x00\x00\x00\x00'
            # context_code = input_ids[idx].detach().cpu().numpy().tobytes()
            
            context_code = input_ids[idx:idx+self.config.prev_n].detach().cpu().numpy().tobytes()
            
            # 如果不忽略历史记录，则将当前的上下文代码添加到历史记录集合中
            if not ignore_history:
                cc_history.add(context_code)
            # 把context_code放到 context_codes列表中
            context_codes = [context_code]
            # 更新哈希对象，先加入上下文代码
            m.update(context_code)
            
            # 得到私钥开始的位置
            # 导入random模块以设置随机种子，确保结果可复现
            import random
            # 设置随机数生成器的种子为42，保证每次运行代码时生成的随机数序列相同
            random.seed(42)
            # 生成一个1024位的随机整数，并将其转换为128字节的大端字节串作为私钥 private_key = b'ke\xa6\xa4\x8b\x81H\xf6\xb3\x8a\x08\x8c\xa6^\xd3\x89\xb7M\x0f\xb12\xe7\x06)\x8f\xad\xc1\xa6\x06\xcb\x0f\xb3\x9a\x1d\xe6D\x81^\xf6\xd1;\x8f\xaa\x187\xf8\xa8\x8b\x17\xfciZ\x07\xa0\xcan\x08"\xe8\xf3l\x03\x11\x99\x97*\x84i\x16A\x9f\x82\x8b\x9d$4\xe4e\xe1P\xbd\x9cf\xb3\xad<-m\x1a=\x1f\xa7\xbc\x89`\xa9#\xb8\xc1\xe99$V\xde>\xb1;\x90FhRW\xbd\xd6@\xfb\x06g\x1a\xd1\x1c\x801\x7f\xa3\xb1y\x9d'
            private_key = random.getrandbits(1024).to_bytes(128, "big")
            # 再加入私钥
            m.update(private_key)
            # 得到私钥结束的位置
                   
            # 获取完整的哈希值 full_hash = b'\x992[6S@\x84\x10z\xfd`\x89d\x0eH\xb3-\xae\'\x8fp\xa9\xed"r\x9f\xf1\x89\xdcZ|\xbe'
            full_hash = m.digest()
            # 从哈希值中提取一个32位的随机种子 seed = 3199385542
            seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
            # 使用列表推导式，对每个上下文代码检查其是否在历史记录中，并获取对应的随机种子
            mask, seeds = zip(
                *[
                    (context_code in cc_history, seed)
                ]
            )
            # 针对每个seed，在对应设备上创建随机数生成器并设置种子  _cdata:1847527152
            rng = [torch.Generator(torch.device("cuda")).manual_seed(seed) for seed in seeds]
            # 根据随机数生成器和scores的列数生成水印代码  # watermark_code = tensor(0.6088, device='cuda:0')
            watermark_code_type = Delta_WatermarkCode(torch.zeros(1)[0])
            watermark_codes = watermark_code_type.from_random(rng, 50272)
            watermark_code = watermark_codes.u[0]
            # 对watermark_code进行裁剪，确保其范围在[0, 1]之间
            watermark_code = torch.clamp(watermark_code, 0, 1)
            # 得到复现的watermark_code 结束的位置
            
            
            # 根据无水印模型生成的logits作为尺子来做检测 核心代码  
            # 拼接：去掉最后5个token的promt + 待检测token的前几个token
            pre_encoding = torch.cat((prompt_cut.to(self.config.device) , input_ids[:idx+self.config.prev_n].to(self.config.device)))
            # 解码成文本
            pre_text = self.config.generation_tokenizer.decode(pre_encoding, skip_special_tokens=True)
            # 对文本进行编码
            pre = self.config.generation_tokenizer(pre_text, return_tensors="pt").to(self.config.device)
            # 禁用梯度计算并获取模型输出
            with torch.no_grad():
                un_watermarked_outputs = self.config.generation_model(**pre)
            # 获取最后一个 token 的 logits，表示预测的概率分布
            un_watermarked_next_logits = un_watermarked_outputs.logits[:, -1, :]
            un_watermarked_cumsum = torch.cumsum(F.softmax(un_watermarked_next_logits, dim=-1), dim=-1)
            
            # 根据不均匀尺子（检测的那个token的概率强行增大到p,例如0.4）来做检测
            p_tensor = torch.full((self.config.vocab_size,), (1-self.config.p) / (self.config.vocab_size - 1))
            p_tensor[input_ids[idx+self.config.prev_n]] = self.config.p
            p_cumsum = torch.cumsum(p_tensor, dim=-1).unsqueeze(0)
            
            # 根据均匀的尺子来做检测
            # 创建一个大小为词汇表大小的全1张量
            tensor_of_ones = torch.ones(self.config.vocab_size).unsqueeze(0)  # tensor_of_ones=[[1., 1., 1.,  ..., 1., 1., 1.]])
            # 使用softmax函数对tensor_of_ones进行归一化处理，并计算其累加和，结果保存在cumsum中  # F.softmax(tensor_of_ones, dim=-1) = [[1.9892e-05, 1.9892e-05, 1.9892e-05,  ..., 1.9892e-05, 1.9892e-05, 1.9892e-05]]
            uniform_cumsum = torch.cumsum(F.softmax(tensor_of_ones, dim=-1), dim=-1)    # cumsum = [[1.9892e-05, 3.9784e-05, 5.9675e-05,  ..., 9.9996e-01, 9.9998e-01, 1.0000e+00]]
            
            # 选择使用哪个尺子
            my_cumsum = un_watermarked_cumsum
            # my_cumsum = p_cumsum
            # my_cumsum = uniform_cumsum
            
            # # 得到(input_ids[idx+1]-1)和input_ids[idx+1]的中位数的值  input_ids[idx]=tensor(5, device='cuda:0')     input_ids[idx+1]=tensor(375, device='cuda:0')
            # curr_value = ( my_cumsum[-1][input_ids[idx+self.config.prev_n]-1].item() + my_cumsum[-1][input_ids[idx+self.config.prev_n]].item() ) / 2    # curr_value = 0.007479312364012003     （当p是2.4时：0.20447574509307742）
            # # 将中位数curr_value转换为张量
            # curr_value_tensor = torch.tensor(curr_value, device='cuda:0')
            # # 对中位数curr_value进行裁剪，确保其范围在[0, 1]之间
            # curr_value_tensor = torch.clamp(curr_value_tensor, 0, 1)
            # # 计算curr_value与水印代码的差值   watermark_code=tensor(0.6088, device='cuda:0')   curr_value_tensor=tensor(0.0016, device='cuda:0')
            # diff = torch.abs( curr_value_tensor - watermark_code )  # diff=tensor(0.6013)   p=0.4时diff=0.4043   使用一半无偏一半kgw时diff=6072
            # diff_200.append(diff)
            
            
            # 使用二分搜索法找到 my_cumsum 中每个code.u元素的位置索引，right=True表示查找大于等于指定值的位置
            index = torch.searchsorted(my_cumsum, watermark_codes.u[..., None], right=True)  # watermark_code.u=[0.6088] index=[[375]]        
            # 对索引进行裁剪，确保其范围在[0, p_logits最后一维长度-1]之间
            index = torch.clamp(index, 0, un_watermarked_next_logits.shape[-1] - 1)
            
            curr_value = input_ids[idx+self.config.prev_n]
            
            if curr_value == index[-1][-1]:
                green_token_count += 1
                # unbiased_watermark_token_count += 1
                green_token_flags.append(1)  # 当前 token 是绿色 token
            else:
                green_token_flags.append(0)  # 当前 token 不是绿色 token 
            
            
            # # 如果在无偏的阈值内 （原来是0.2）
            # if diff < self.config.gamma:
            #     green_token_count += 1
            #     unbiased_watermark_token_count += 1
            #     green_token_flags.append(1)  # 当前 token 是绿色 token
            # else:
            #     # 如果不在无偏的阈值内  prefix_length
            #     if idx < self.config.prefix_length:
            #         green_token_flags.append(0)  # 当前 token 不是绿色 token 
            #     else:    
            #         # 获取当前 token  tensor(353, device='cuda:0')
            #         curr_token = input_ids[idx+self.config.prev_n]
            #         # 获取当前 token 的绿色列表
            #         greenlist_ids = self.get_greenlist_ids(input_ids[:idx+self.config.prev_n])
            #         # 检查当前 token 是否在绿色列表中  true
            #         if curr_token in greenlist_ids:
            #             green_token_count += 1
            #             kgw_watermark_token_count += 1
            #             green_token_flags.append(1)  # 当前 token 是绿色 token
            #         else:
            #             green_token_flags.append(0)  # 当前 token 不是绿色 token
                
        # 将张量从CUDA转移到CPU计算diffs的平均值（一句话中的200个token与200个watermark_code的差值的平均值）
        # diff_avg = torch.mean(torch.stack([diff.cpu() for diff in diff_200]))
        # print(diff_avg.item())
        # diff_avg.unsqueeze(0)
        # return diff_avg.item(), green_token_flags  
        
        # 计算 z 分数
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags  
    #add end

# add by lihe
from abc import ABC, abstractmethod
from torch.nn import functional as F
from torch import FloatTensor, LongTensor
from dataclasses import dataclass, field

class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random(
        cls,
        #  rng: Union[torch.Generator, list[torch.Generator]],
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        """When rng is a list, it should have the same length as the batch size."""
        pass

class Delta_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, u: FloatTensor):
        assert torch.all(u >= 0) and torch.all(u <= 1)
        self.u = u

    @classmethod
    def from_random(
        cls,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            u = torch.stack(
                [
                    torch.rand((), generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            u = torch.rand((), generator=rng, device=rng.device)
        return cls(u)
# add end


# add by lihe
class AbstractReweight(ABC):
    watermark_code_type: type[AbstractWatermarkCode]

    @abstractmethod
    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        pass
 
# 定义一个名为Delta_Reweight的类，继承自AbstractReweight基类
class Delta_Reweight(AbstractReweight):
    # 类属性，指定水印代码类型为Delta_WatermarkCode
    watermark_code_type = Delta_WatermarkCode

    # 重写__repr__方法，用于返回类的字符串表示形式
    def __repr__(self):
        return f"Delta_Reweight()"  # 返回类的名称作为字符串

    # 定义一个方法reweight_logits，用于对logits进行重新加权 #p_logits.argmax()=78 p_logits [[-4.4564, -4.2499,    -inf,  ..., -4.5518, -4.2840, -4.5727]]
    def reweight_logits(
        self, watermark_code: AbstractWatermarkCode, p_logits: FloatTensor  # 输入参数：水印代码对象和原始logits张量，形状为[1, 50264]
    ) -> FloatTensor:  # 返回类型为FloatTensor
        
        # add by wang
        # if on:
        #     mask = p_logits != ('-inf')
        #     p_logits[mask] = 0.02
        # add end
        
        # 使用softmax函数对p_logits进行归一化处理，并计算其累加和，结果保存在cumsum中  p_logits.argmax() = tensor(78, device='cuda:0')  p_logits.max()=tensor(13.0609, device='cuda:0') F.softmax(p_logits, dim=-1).max()=tensor(0.2744, device='cuda:0')
        cumsum = torch.cumsum(F.softmax(p_logits, dim=-1), dim=-1)          
        # 使用二分搜索法找到cumsum中每个code.u元素的位置索引，right=True表示查找大于等于指定值的位置
        index = torch.searchsorted(cumsum, watermark_code.u[..., None], right=True)  # watermark_code.u=[0.6088] index=[[375]]        
        # 对索引进行裁剪，确保其范围在[0, p_logits最后一维长度-1]之间
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        # 创建两个与p_logits同形状的张量，一个全为0，一个全为负无穷大
        zeros_like_logits = torch.full_like(p_logits, 0)
        neg_inf_like_logits = torch.full_like(p_logits, float("-inf"))
        # 使用torch.where根据条件构建新的logits张量
        # 条件为：将索引位置的元素设置为0，其余位置设置为负无穷大
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device) == index,  # 条件判断
            zeros_like_logits,  # 条件为真时的值
            neg_inf_like_logits,  # 条件为假时的值
        )
        
        return modified_logits  # 返回修改后的logits张量 #[[-inf, -inf, -inf,  ..., -inf, -inf, -inf]] 只有一个位置有值
# add end


# add by lihe    
class AbstractContextCodeExtractor(ABC):
    @abstractmethod
    def extract(self, context: LongTensor) -> any:
        """Should return a context code `c` which will be used to initialize a torch.Generator."""
        pass

@dataclass
class All_ContextCodeExtractor(AbstractContextCodeExtractor):
    def extract(self, context: LongTensor) -> any:
        return context.detach().cpu().numpy().tobytes()

@dataclass
class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    """Extracts the last n tokens in the context"""

    n: int

    def extract(self, context: LongTensor) -> any:
        return context[-self.n :].detach().cpu().numpy().tobytes()

# add end


class KGWLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for KGW algorithm, process logits to add watermark."""

    def __init__(self, config: KGWConfig, utils: KGWUtils, private_key: any, reweight: AbstractReweight, context_code_extractor: AbstractContextCodeExtractor, ignore_history=False, *args, **kwargs) -> None:
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
        # 保存私钥对象
        self.private_key = private_key
        # 保存重新加权对象
        self.reweight = reweight
        # 保存上下文代码提取器对象
        self.context_code_extractor = context_code_extractor
        # 保存是否忽略历史对象
        self.ignore_history = ignore_history
        # 初始化一个集合，用于存储已处理过的上下文代码
        self.cc_history = set()

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

    # add by lihe 
    # 定义一个方法，根据上下文代码获取随机数种子
    def get_rng_seed(self, context_code: any) -> any:
        # 如果不忽略历史记录，则将当前的上下文代码添加到历史记录集合中
        if not self.ignore_history:
            self.cc_history.add(context_code)
        # 导入hashlib模块，用于计算SHA-256哈希
        import hashlib
        # 创建一个SHA-256哈希对象
        m = hashlib.sha256()
        # 更新哈希对象，先加入上下文代码
        m.update(context_code)
        # 再加入私钥
        m.update(self.private_key)
        # 获取完整的哈希值 full_hash = b'4\xfbh\x91\xa3\xe9\x07\x19\x9bVT\xdb\x00PQ3c\x19\xc8\xb3\x8f[\xdaq\xd1\xd3\x02\x06\x85\xdf\x04\xe1'
        full_hash = m.digest()
        # 从哈希值中提取一个32位的随机种子  seed = 3199385542
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        # 返回随机种子
        return seed
    # add end
    
    # add by lihe 
    # 定义一个内部使用的方法，根据输入的ids获取上下文代码及其是否在历史中的标记和对应的随机种子
    def _get_codes(self, input_ids: LongTensor):
        # 获取输入序列的批次大小 input_ids.size()=[1,30]
        batch_size = input_ids.size(0)
        # 遍历批次，对每个样本调用上下文代码提取器并收集结果 context_codes = [b'\x05\x00\x00\x00\x00\x00\x00\x00']
        context_codes = [
            self.context_code_extractor.extract(input_ids[i]) for i in range(batch_size)
        ]
        # 使用列表推导式，对每个上下文代码检查其是否在历史记录中，并获取对应的随机种子
        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self.get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )
        # 返回上下文代码的历史存在标记和对应的随机种子
        return mask, seeds
    # add end

    # add by lihe 
    # 定义_core辅助方法，输入为input_ids和scores张量，用于内部处理
    def _core(self, input_ids: LongTensor, scores: FloatTensor):
        # 调用_get_codes方法根据input_ids获取掩码mask和种子seeds
        mask, seeds = self._get_codes(input_ids) 
        # 针对每个seed，在对应设备上创建随机数生成器并设置种子
        rng = [torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds]
        # 将掩码mask转换为与scores相同设备上的张量
        mask = torch.tensor(mask, device=scores.device)
        # 使用reweight模块，根据随机数生成器和scores的列数生成水印代码watermark_code u = [0.6088]
        watermark_code = self.reweight.watermark_code_type.from_random(rng, scores.size(1))
        
        
        # 根据watermark_code进行德尔塔加权
        reweighted_scores = self.reweight.reweight_logits(watermark_code=watermark_code, p_logits=scores)
  
        # 返回掩码和重新加权后的得分
        return mask, reweighted_scores
    # add end

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        # 如果输入序列长度小于前缀长度，则直接返回原始 scores #input_ids.shape=[1, 30]
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores
        # add by lihe
        # 调用_core方法处理input_ids和scores，获得掩码和重新加权的得分
        mask, reweighted_scores = self._core(input_ids, scores)
        # 如果忽略历史（ignore_history为True），直接返回重新加权的得分
        if self.ignore_history:
            return reweighted_scores
        else:
            # 否则，根据掩码应用scores或reweighted_scores，使用where函数实现条件选择
            return torch.where(mask[:, None], scores, reweighted_scores)
        # add end   
    
    
    # 原来的代码
    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        # 对 greenlist_mask 中为 True 的位置的 scores 值增加 greenlist_bias
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias #bias=2.0    score的维度是[1, 50272]   scores[greenlist_mask]的维度是[25136]
        return scores     
    
class White_Box(BaseWatermark):
    """Top-level class for KGW algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, ignore_history=False, *args, **kwargs) -> None:
        """
            Initialize the KGW algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        
        # add by lihe
        if algorithm_config is None:
            config_dict = load_config_file('config/KGW.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'KGW':
            raise AlgorithmNameMismatchError('KGW', config_dict['algorithm_name'])
        
        # 设置随机数生成器的种子为42，保证每次运行代码时生成的随机数序列相同
        random.seed(42)
        # 生成一个1024位的随机整数，并将其转换为128字节的大端字节串作为私钥
        private_key = random.getrandbits(1024).to_bytes(128, "big")
        # 使用私钥、Delta重权策略和前1个上下文码提取器,创建WatermarkLogitsProcessor实例
        self.private_key = private_key
        self.reweight = Delta_Reweight()
        self.context_code_extractor = PrevN_ContextCodeExtractor(config_dict['prev_n'])
        self.ignore_history = config_dict['ignore_history']
        # add end 
        
        # 初始化算法配置
        self.config = KGWConfig(algorithm_config, transformers_config)
        # 初始化工具类，用于辅助方法
        self.utils = KGWUtils(self.config)
        # 初始化 logits 处理器，用于生成带水印的文本
        self.logits_processor = KGWLogitsProcessor(self.config, self.utils, private_key = self.private_key, reweight = self.reweight, context_code_extractor = self.context_code_extractor, ignore_history = self.ignore_history)
        
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark 配置生成带水印文本的方法
        generate_with_watermark = partial(
            self.config.generation_model.generate,  # 调用生成模型的生成方法
            logits_processor=LogitsProcessorList([self.logits_processor]),   # 设置 logits 处理器列表
            **self.config.gen_kwargs  # 传递生成配置参数
        )
        
        # Encode prompt # 对输入提示进行编码 
        '''[[    2,  9325, 13010,  2113,     6,  2869, 31126,  5371,  2013,   118,
                34,  2208,   129,  2144,  4893,     4,  1646,  4963,    36, 18591,
                195,  6668,     4,   466,   153,    43,    25,   903,   148,     5]]
        '''
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # Generate watermarked text # 生成带水印的文本
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode  将生成的编码文本解码为字符串
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    # 核心代码，后续需要把prompt参数去掉
    def detect_watermark(self, text: str, prompt_cut, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text # 对输入文本进行编码 # encoded_text.shape = [201]
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # 原来的代码
        # Compute z_score using a utility method  # 计算 z_score，用于检测水印
        # z_score = self.utils.score_sequence(encoded_text)
        # Determine if the z_score indicates a watermark  # 根据 return_dict 参数决定返回结果的格式
        # is_watermarked = z_score > self.config.z_threshold
        # Return results based on the return_dict flag
        # if return_dict:
        #     return {"is_watermarked": is_watermarked, "score": z_score}  # 返回字典格式的结果
        # else:
        #     return (is_watermarked, z_score)  # 返回元组格式的结果
        
        # add by lihe 
        # 计算每句话的平均差值
        # diff_avg, _ = self.utils.score_sequence(encoded_text, prompt_cut)
        # print(diff_avg)
        # 根据阈值判断是否有水印 核心代码 原来是0.2
        # is_watermarked = (diff_avg < self.config.gamma)
        
        # 计算 z_score，用于检测水印  
        z_score, _ = self.utils.score_sequence(encoded_text, prompt_cut)
        
        # Determine if the z_score indicates a watermark  
        is_watermarked = z_score > self.config.z_threshold
        
        # 设置返回结果的格式 # 根据 return_dict 参数决定返回结果的格式
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}  # 返回字典格式的结果
        else:
            return (is_watermarked, z_score)  # 返回元组格式的结果
    
        # add end
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text  对输入文本进行编码
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Compute z-score and highlight values # 计算 z-score 和高亮值，用于可视化
        z_score, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens 解码单个 token
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())  # 解码 token ID 为字符串
            decoded_tokens.append(token)
        # 返回解码后的 token 列表和高亮值列表，用于可视化
        return DataForVisualization(decoded_tokens, highlight_values)