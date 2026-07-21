from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn
import torch.nn.functional as F


class AbstractCellTokenizer(nn.Module, ABC):
    """Base class for all cell tokenizers.

    Global contract:
    - input shape: [M, 2, H, W]
    - output shape: [M, Nt, Lt]

    Design split:
    - `Nt` and `Lt` are fixed by the config used to instantiate the tokenizer
    - `H` and `W` are fixed by the concrete realization itself
    """

    def __init__(self, *, num_tokens: int, token_dim: int, output_norm: str = 'batchnorm') -> None:
        super().__init__()
        if int(num_tokens) <= 0:
            raise ValueError(f'num_tokens must be positive, got {num_tokens}')
        if int(token_dim) <= 0:
            raise ValueError(f'token_dim must be positive, got {token_dim}')
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.output_norm = str(output_norm).strip().lower()
        if self.output_norm in {'batchnorm', 'bn'}:
            self.output_norm = 'batchnorm'
            self.output_batch_norm = nn.BatchNorm1d(self.num_tokens * self.token_dim, affine=False)
            self.output_layer_norm = None
        elif self.output_norm in {'layernorm', 'ln'}:
            self.output_norm = 'layernorm'
            self.output_batch_norm = None
            self.output_layer_norm = nn.LayerNorm(self.token_dim, elementwise_affine=False)
        elif self.output_norm in {'none', 'identity'}:
            self.output_norm = 'none'
            self.output_batch_norm = None
            self.output_layer_norm = None
        else:
            raise ValueError(f'Unsupported tokenizer output_norm: {output_norm!r}')

    @property
    def Nt(self) -> int:
        return self.num_tokens

    @property
    def Lt(self) -> int:
        return self.token_dim

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input_shape(x)
        self.validate_realization_input(x)
        tokens = self.forward_tokens(x)
        self._validate_output_shape(tokens, batch_size=x.shape[0])
        if self.output_norm == 'batchnorm':
            flat_tokens = tokens.reshape(tokens.shape[0], -1)
            normalized = self.output_batch_norm(flat_tokens)
            return normalized.view_as(tokens)
        if self.output_norm == 'layernorm':
            return self.output_layer_norm(tokens)
        return tokens

    @abstractmethod
    def forward_tokens(self, x: Tensor) -> Tensor:
        """Implement the tokenizer body.

        Args:
            x: input tensor with shape [M, 2, H, W]

        Returns:
            Tensor with shape [M, Nt, Lt]
        """
        raise NotImplementedError

    def validate_realization_input(self, x: Tensor) -> None:
        """Optional realization-specific input validation.

        Concrete tokenizers should override this when they require fixed `H/W`.
        """
        return None

    def _validate_input_shape(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f'Expected tokenizer input shape [M, 2, H, W], got {tuple(x.shape)}')
        if x.shape[1] != 2:
            raise ValueError(f'Tokenizer expects exactly 2 input channels, got shape {tuple(x.shape)}')

    def _validate_output_shape(self, tokens: Tensor, *, batch_size: int) -> None:
        expected = (batch_size, self.Nt, self.Lt)
        if tuple(tokens.shape) != expected:
            raise ValueError(f'Tokenizer must output shape {expected}, got {tuple(tokens.shape)}')
