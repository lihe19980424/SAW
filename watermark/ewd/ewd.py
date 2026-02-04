# ============================================
# ewd.py
# Description: Implementation of EWD algorithm
# ============================================

import torch
from functools import partial
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from visualize.data_for_visualization import DataForVisualization


class EWDConfig:
    """Config class for EWD algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EWD configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/EWD.json')
        else:
            config_dict = algorithm_config
        if config_dict['algorithm_name'] != 'EWD':
            raise AlgorithmNameMismatchError('EWD', config_dict['algorithm_name'])

        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        
        self.temperature_inner = config_dict['temperature_inner']
        
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs
        
        print("algorithm_name: EWD", " temperature_inner", self.temperature_inner)


class EWDUtils:
    """Utility class for EWD algorithm, contains helper functions."""

    def __init__(self, config: EWDConfig, *args, **kwargs) -> None:
        """
            Initialize the EWD utility class.

            Parameters:
                config (EWDConfig): Configuration for the EWD algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        alpha = torch.exp(torch.tensor(self.config.delta)).item()
        self.z_value = ((1 - self.config.gamma) * (alpha - 1))/(1 - self.config.gamma + (alpha * self.config.gamma))

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last token in the input_ids."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids
    
    def calculate_entropy(self, model, tokenized_text) -> list[float]:
        """Calculate the entropy of the tokenized text using the model."""
        with torch.no_grad():
            # =========== 新增：安全检查与截断 ===========
            # 获取模型 Embedding 层的最大容量 (即词表大小)
            if hasattr(model, "get_input_embeddings"):
                embedding_layer = model.get_input_embeddings()
                if embedding_layer is not None:
                    vocab_size = embedding_layer.weight.shape[0]
                    # 如果发现有 Token ID 超出范围，将其截断为 vocab_size - 1 (通常是 unknown 或最后一个有效 token)
                    if torch.max(tokenized_text) >= vocab_size:
                        # print(f"Warning: Detected out-of-bounds token IDs. Clamping to {vocab_size - 1}.")
                        tokenized_text = torch.clamp(tokenized_text, max=vocab_size - 1)
            # ==========================================
            
            # 获取输入张量 [1, seq_len]
            input_tensor = torch.unsqueeze(tokenized_text, 0)
            
            # 【修改点】判断模型是否为 Encoder-Decoder 架构 (如 NLLB, T5, BART)
            if model.config.is_encoder_decoder:
                # 对于 Seq2Seq 模型，必须提供 decoder_input_ids。
                # 由于在检测阶段我们只有生成的文本（target），没有原始的 prompt（source），
                # 为了让代码跑通并计算自身的熵，我们将 input_ids 和 decoder_input_ids 都设为这段文本。
                output = model(input_ids=input_tensor, decoder_input_ids=input_tensor, return_dict=True)
            else:
                # 对于 Decoder-only 模型 (如 LLaMA, GPT, OPT)
                output = model(input_tensor, return_dict=True)
            
            # output = model(input_ids=torch.unsqueeze(tokenized_text, 0), return_dict=True)
            probs = torch.softmax(output.logits, dim=-1)
            denoms = 1+(self.z_value * probs)
            renormed_probs = probs / denoms
            sum_renormed_probs = renormed_probs.sum(dim=-1)
            entropy=sum_renormed_probs[0].cpu().tolist()
            entropy.insert(0, -10000.0)
            return entropy[:-1]
    
    def _get_weight_from_entropy(self, entropy_list: list[float]) -> list:
        """Compute the weights from the entropy list."""

        # Convert the entropy list to a tensor
        entropy_tensor = torch.tensor(entropy_list)

        # Compute the minimum entropy from the elements beyond the prefix length
        min_entropy = torch.min(entropy_tensor[self.config.prefix_length:])

        # Subtract this minimum entropy from all entropy values
        adjusted_entropies = entropy_tensor - min_entropy

        # Create a list where the prefix part is filled with -1, and the rest with adjusted entropies
        weights = [-1] * self.config.prefix_length + adjusted_entropies[self.config.prefix_length:].tolist()

        return weights
    
    def _compute_z_score(self, observed_count: float, weight: list) -> float:
        """Compute the z-score for the given observed count and weight."""

        weight_tensor = torch.tensor(weight, dtype=torch.float)
        expected_count = self.config.gamma
        numer = observed_count - expected_count * torch.sum(weight_tensor)
        denom = torch.sqrt(torch.sum(torch.square(weight_tensor)) * expected_count * (1 - expected_count))
        z = numer / denom
        return z.item()

    def score_sequence(self, input_ids: torch.Tensor, entropy_list) -> tuple[float, list[int], list]:
        """Score the input_ids using the entropy list."""

        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            # raise ValueError(
            #     (
            #         f"Must have at least {1} token to score after "
            #     )
            # )
            green_token_flags = [-1 for _ in range(self.config.prefix_length)]
            return -10.0, green_token_flags, [-1]

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]
        
        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)

        # calculate weights
        weights = self._get_weight_from_entropy(entropy_list)

        # sum up weights where green_token_flags = 1 to get green_token_count
        green_token_count = sum(weights[i] for i in range(len(green_token_flags)) if green_token_flags[i] == 1)

        # compute z_score
        z_score = self._compute_z_score(green_token_count, weights[self.config.prefix_length:])
        return z_score, green_token_flags, weights


class EWDLogitsProcessor(LogitsProcessor):
    """Logits processor for EWD algorithm, contains logits processing functions."""

    def __init__(self, config: EWDConfig, utils: EWDUtils, *args, **kwargs) -> None:
        """
            Initialize the EWD logits processor.

            Parameters:
                config (EWDConfig): Configuration for the EWD algorithm.
                utils (EWDUtils): Utility functions for the EWD algorithm.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        # scores = scores / self.config.temperature_inner
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores


class EWD(BaseWatermark):
    """Top-level class for EWD algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EWD algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = EWDConfig(algorithm_config, transformers_config)
        self.utils = EWDUtils(self.config)
        self.logits_processor = EWDLogitsProcessor(self.config, self.utils)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt, temperature=self.config.temperature_inner, pad_token_id=self.config.generation_tokenizer.eos_token_id)
        # decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in text."""
        if len(text) == 0:
            # raise ValueError("Text cannot be empty")
            print("Text is empty")
            z_score = -10.0
            is_watermarked = z_score > self.config.z_threshold

            # Return results based on the return_dict flag
            if return_dict:
                return {"is_watermarked": is_watermarked, "score": z_score}
            else:
                return (is_watermarked, z_score)

        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # compute z_score
        z_score, _, _ = self.utils.score_sequence(encoded_text, entropy_list)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def get_data_for_visualization(self, text: str, *args, **kwargs):
        """Get data for visualization."""
        
        # encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # calculate entropy
        entropy_list = self.utils.calculate_entropy(self.config.generation_model, encoded_text)
        
        # compute z-score, highlight values, and weights
        z_score, highlight_values, weights = self.utils.score_sequence(encoded_text, entropy_list)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values, weights)