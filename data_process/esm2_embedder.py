import warnings
from typing import List, Generator, Union, Iterable, Optional, Any, Tuple
from itertools import tee

import torch
from esm.pretrained import load_model_and_alphabet  # ✅ 确保导入正确
from numpy import ndarray

from bio_embeddings.embed import EmbedderInterface
from bio_embeddings.utilities import get_model_file


def load_model_and_alphabet_local(model_location: str) -> Tuple[Any, Any]:
    """Custom bio_embeddings versions because we change names and don't have regression weights"""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Regression weights not found, predicting contacts will not produce correct results.",
    )

    model_data = torch.load(model_location, map_location="cpu")
    return load_model_and_alphabet(model_data, None)


class ESMEmbedderBase(EmbedderInterface):
    embedding_dimension = 1280  # 维度，ESM2 可能更高
    number_of_layers = 1
    necessary_files = ["model_file"]
    max_len = 1022  # ESM 的 max_len，ESM2 可能更长

    _picked_layer: int

    def __init__(self, device: Union[None, str, torch.device] = None, **kwargs):
        super().__init__(device, **kwargs)

    def embed(self, sequence: str) -> ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        batch, batch_copy = tee(batch)
        data = [(str(pos), sequence) for pos, sequence in enumerate(batch)]
        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)

        with torch.no_grad():
            results = self._model(
                batch_tokens.to(self._device), repr_layers=[self._picked_layer]
            )
        token_embeddings = results["representations"][self._picked_layer]

        for i, (_, seq) in enumerate(data):
            yield token_embeddings[i, 1 : len(seq) + 1].cpu().numpy()




class ESM2Embedder(EmbedderInterface):
    """
    ESM-2 Embedder（基于 Hugging Face Transformers 加载）

    参考文献：
      Lin, Zeming, et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model."
      Science 379.6629 (2023). https://doi.org/10.1126/science.ade2574
    """
    name = "esm2"
    embedding_dimension = 1280
    # 注意：实际模型可能有更长的序列上限，这里设置一个适当的值
    max_len = 1024

    def __init__(
        self,
        model_version: str = "facebook/esm2_t33_650M_UR50D",
        device: Union[None, str, torch.device] = None,
        **kwargs,
    ):
        self.model_version = model_version
        self._device = torch.device(device) if device is not None else torch.device("cpu")

        # 这里使用 Hugging Face Transformers 加载模型及分词器
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        # 使用 AutoModel 以获取隐藏状态（确保 output_hidden_states=True）
        self._model = AutoModel.from_pretrained(
            self.model_version, output_hidden_states=True
        ).to(self._device)

    def embed(self, sequence: str) -> ndarray:
        [embedding] = list(self.embed_batch([sequence]))
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """
        对于每条输入序列，先使用分词器编码，再前向计算获取 token embedding，
        最后对 token embedding 进行平均得到序列 embedding。
        """
        for sequence in batch:
            if len(sequence) > self.max_len:
                raise ValueError(
                    f"序列长度 {len(sequence)} 超过了 {self.name} 允许的最大长度 {self.max_len}"
                )
            inputs = self._tokenizer(sequence, return_tensors="pt")
            # 将输入转移到指定设备上
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # 获取最后一层隐藏状态，形状为 (batch_size, seq_len, hidden_dim)
            # 对于 ESM2，通常不会加入额外的特殊 token，但如果分词器加入了 CLS/EOS，
            # 可通过以下判断去除首尾特殊 token：
            token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
            input_ids = inputs["input_ids"][0]
            # 如果存在 CLS token，则去除第一个 token（以及可能存在的 EOS token）
            if self._tokenizer.cls_token_id is not None and input_ids[0] == self._tokenizer.cls_token_id:
                token_embeddings = token_embeddings[:, 1:]
                if self._tokenizer.eos_token_id is not None and input_ids[-1] == self._tokenizer.eos_token_id:
                    token_embeddings = token_embeddings[:, :-1]

            # 对 token 进行平均（也可以根据需求采用其他聚合方式）
            embedding = token_embeddings.cpu().numpy().squeeze(0)
            yield embedding

    def embed_many(
        self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        batch = []
        for seq in sequences:
            batch.append(seq)
            if batch_size is not None and len(batch) == batch_size:
                yield from self.embed_batch(batch)
                batch = []
        if batch:
            yield from self.embed_batch(batch)

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:

        return embedding









if __name__ == "__main__":
    embedder = ESM2Embedder()  # 确保正确的模型名称
    test_sequence = "MKTLLLTLVVVTIVCLDLGYT"  # 你的氨基酸序列
    embedding = embedder.embed(test_sequence)

    print("Embedding shape:", embedding.shape)  # 查看 embedding 维度

