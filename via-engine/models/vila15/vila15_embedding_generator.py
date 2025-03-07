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

import concurrent.futures
import os
import sys

import numpy as np
import tensorrt as trt
import torch
from transformers import AutoConfig, AutoModel

from chunk_info import ChunkInfo
from via_logger import TimeMeasure, logger

sys.path.append(os.path.dirname(__file__) + "/VILA")

import llava.model.language_model.llava_llama  # noqa: F401, E402


def trt_dtype_to_torch(dtype):
    """Translate TRT datatype to torch data type"""
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


class Vila15EmbeddingGenerator:
    """Visual Embedding Generator for the VILA 1.5 model"""

    def __init__(
        self, model_path: str, use_trt=False, trt_engine_dir="", async_output=False
    ) -> None:
        """Vila15EmbeddingGenerator initializer

        Args:
            model_path: Path where the model is located
            use_trt: Use TRT to generate the embeddings. Defaults to False.
            trt_engine_dir: Path to the directory where the TRT engines for the model are located.
                            Defaults to "".
        """
        self._use_trt = use_trt
        self._config = AutoConfig.from_pretrained(model_path)
        if self._use_trt:
            # TRT mode
            from tensorrt_llm.runtime import Session

            with TimeMeasure("VILA Embeddings TRT Model load"):
                # Load TRT model from serialized engine
                vision_encoder_path = os.path.join(
                    trt_engine_dir, "visual_engines", "visual_encoder.engine"
                )
                logger.info(f"Loading engine from {vision_encoder_path}")
                with open(vision_encoder_path, "rb") as f:
                    engine_buffer = f.read()
                logger.info(f"Creating session from engine {vision_encoder_path}")
                self.visual_encoder_session = Session.from_serialized_engine(engine_buffer)

                # Load layers that are required for additional processing after
                # passing the frames through TRT engine
                device_map = {
                    "model.vision_tower": "meta",
                    "model.embed_tokens": "cuda",
                    "model.layers": "meta",
                    "model.norm": "meta",
                    "lm_head": "meta",
                    "model.mm_projector": "meta",
                }
                self._model = AutoModel.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    # torch_dtype=torch.float16,
                )
                self.stream = torch.cuda.Stream(torch.cuda.current_device())
                torch.cuda.set_stream(self.stream)
            self._output_tpool = (
                concurrent.futures.ThreadPoolExecutor(max_workers=2) if async_output else None
            )
        else:
            # PyTorch mode
            with TimeMeasure("VILA Embeddings Model load"):
                # Load layers that are required for embedding generation
                device_map = {
                    "model.embed_tokens": "cuda",
                    "model.layers": "meta",
                    "model.norm": "meta",
                    "lm_head": "meta",
                }
                self._model = AutoModel.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                )
                self._mm_proj = self._model.mm_projector.to(device="cuda", dtype=torch.float16)
                # Get the vision tower from the model
                vision_tower = self._model.get_vision_tower()
                vision_tower.to(device="cuda", dtype=torch.float16)

    def warmup(self):
        input_dims = self.visual_encoder_session._engine.get_tensor_profile_shape("input", 0)[-1]
        input_dims = [int(d) for d in input_dims[:4]]
        frame_input = torch.zeros(size=input_dims, dtype=torch.float16, device="cuda")
        self.get_embeddings(frame_input.unsqueeze(0))

    def get_embeddings(self, frames_tensor_batch: list):
        """Get embeddings for a batch of chunks. For each chunk a list of frames is needed.

        Args:
            frames_list_batch (list): List of list of frames per chunk

        Returns:
            List of embeddings tensor for all input chunks
        """
        with TimeMeasure("VILA Embeddings generation"):
            if self._use_trt:
                visual_outputs_batch = []
                for frames_tensor in frames_tensor_batch:
                    # TRT mode
                    from tensorrt_llm.runtime import TensorInfo

                    visual_output_info = self.visual_encoder_session.infer_shapes(
                        [TensorInfo("input", trt.DataType.HALF, frames_tensor.shape)]
                    )
                    visual_outputs = {
                        t.name: torch.empty(
                            tuple(t.shape[:3]),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device=frames_tensor.device,
                        )
                        for t in visual_output_info
                    }
                    ok = self.visual_encoder_session.run(
                        {"input": frames_tensor}, visual_outputs, self.stream.cuda_stream
                    )
                    assert ok, "Runtime execution failed for vision encoder session"
                    visual_outputs_batch.append(visual_outputs["output"])
                self.stream.synchronize()

                return visual_outputs_batch
            else:
                embeds = [
                    self._mm_proj(self._model.get_vision_tower()(frames_tensor))
                    for frames_tensor in frames_tensor_batch
                ]
                return embeds


if __name__ == "__main__":
    import argparse

    from vlm_pipeline.video_file_frame_getter import (
        DefaultFrameSelector,
        VideoFileFrameGetter,
    )

    parser = argparse.ArgumentParser(description="VILA 1.5 Embedding Generator")
    parser.add_argument("file", type=str, help="File to generate the embeddings for")
    parser.add_argument("out", type=str, help="File to dump the embeddings to")

    parser.add_argument(
        "--start-time", type=int, default=0, help="Start time in sec to get frames from"
    )

    parser.add_argument(
        "--end-time", type=int, default=-1, help="End time in sec to get frames from"
    )
    parser.add_argument(
        "--use-trt", action="store_true", help="Use TensorRT for generating embeddings"
    )

    args = parser.parse_args()

    chunk = ChunkInfo()
    chunk.file = args.file
    chunk.start_pts = args.start_time * 1000000000
    chunk.end_pts = args.end_time * 1000000000 if args.end_time >= 0 else -1

    frame_getter = VideoFileFrameGetter(
        DefaultFrameSelector(100),
        model_path="/home/ubuntu/llava-v1.5-sharegpt-ptft-13b-lita-soft-ce-im-vid-se",
    )
    frames, frames_pts = frame_getter.get_frames(chunk)

    embgen = Vila15EmbeddingGenerator(
        "/home/ubuntu/llava-v1.5-sharegpt-ptft-13b-lita-soft-ce-im-vid-se",
        args.use_trt,
        "/home/ubuntu/llava-v1.5-sharegpt-ptft-13b-lita-soft-ce-im-vid-se/trt-engines/fp16/0-gpu/",
    )
    emb = embgen.get_embeddings([frames])[0]

    np.savetxt(args.out, emb.cpu().detach())
