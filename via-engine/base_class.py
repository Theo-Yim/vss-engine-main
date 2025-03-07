################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class VlmGenerationConfig:
    temperature: float = 0.2
    maxNewTokens: int = 1024
    topP: float = 1.0
    seed: int = 1


class EmbeddingGeneratorBase(ABC):

    @abstractmethod
    def get_embeddings(self, frames: List[List[torch.tensor]]) -> list[torch.Tensor]:
        pass


class CustomModelBase(ABC):

    @abstractmethod
    def get_embedding_generator(self) -> EmbeddingGeneratorBase:
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        input_tensors: List[torch.Tensor],
        video_frames_times: List[List],
        generation_config: VlmGenerationConfig,
    ):
        pass
