from snip.envs.environment import FunctionEnvironment
from argparse import Namespace
# from omegaconf import OmegaConf
import torch

class Tokenizer:
    def __init__(self, params: Namespace, max_len: int = 200):
        self.max_len = max_len
        self.env = FunctionEnvironment(params, None)
        self.vocab_size = len(self.env.equation_id2word)
        self.pad_token_id = self.env.equation_word2id["<PAD>"]
        self.eos_token_id = self.env.equation_word2id["<EOS>"]
        
        # Add BOS if not present
        # if "[BOS]" not in self.env.equation_word2id:
        #     bos_idx = len(self.env.equation_word2id)
        #     self.env.equation_word2id["[BOS]"] = bos_idx
        #     self.env.equation_id2word[bos_idx] = "[BOS]"
        #     self.vocab_size += 1
        # else:
        #     bos_idx = self.env.equation_word2id["[BOS]"]

        # self.bos_token_id = bos_idx
        
        
    def tokenize_to_index(self, x: list[str]) -> torch.Tensor:
        decoded = self.env.equation_encoder.decode(x)
        encoded = self.env.equation_encoder.encode(decoded)
        return self.env.word_to_idx([encoded], float_input=False)[0]
    
    def encode(self, x: list[str]) -> torch.Tensor:
        indices = self.tokenize_to_index(x)
        eos = torch.tensor([self.eos_token_id])
        indices = torch.cat([indices, eos], dim=0)
        
        if len(indices) < self.max_len:
            pad_list = torch.tensor([self.pad_token_id] * (self.max_len-len(indices)))
            indices = torch.cat([indices, pad_list], dim=0)
        elif len(indices) > self.max_len:
            indices = indices[:self.max_len-1]
            indices = torch.cat([indices, eos], dim=0)
        
        return indices
    
    def __call__(self, xs: str | list[str] | list[list[str]]) -> torch.Tensor:
        if isinstance(xs, str):
            return self.encode(xs.split(","))
        elif isinstance(xs, list):
            return self.encode(xs)
        
        return torch.stack([self.encode(x) for x in xs])
    
    
    def __len__(self):
        return self.vocab_size
    
    def decode_infix(self, x: list[int] | torch.Tensor) -> str:
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        
        return self.env.idx_to_infix(x, is_float=False)
    
    
    def batch_decode(self, xs: list[int] | list[list[int]] | torch.Tensor) -> list[str]:
        if isinstance(xs, torch.Tensor):
            if xs.dim() == 1:
                return [self.decode_infix(xs)]
            return [self.decode_infix(x) for x in xs]
        
        if isinstance(xs, list[int]):
            return [self.decode_infix(xs)]
        return [self.decode_infix(x) for x in xs]
    
    
        
        