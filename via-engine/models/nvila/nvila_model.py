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
import random
import re
import sys

import numpy
import torch
from filelock import FileLock

from via_logger import TimeMeasure


class NVila:

    def __init__(self, model_path, max_batch_size=None, use_trt=False, **kwargs) -> None:
        self._model = None
        self._max_batch_size = max_batch_size
        self._inflight_req_ids = []
        self._use_trt = use_trt

        if bool(os.environ.get("NVILA_USE_PYTORCH", "")):
            self._use_trt = False

        if self._use_trt:
            # TRT Model
            with TimeMeasure("VILA TRT model load"):
                # Load the TRT model
                import tensorrt_llm.bindings
                from tensorrt_llm._torch import LLM

                if "NVILA_VIDEO_MAX_TILES" not in os.environ:
                    os.environ["NVILA_VIDEO_MAX_TILES"] = "4"

                with FileLock(model_path + "/.lock"):
                    self._llm = LLM(
                        model=model_path,
                        kv_cache_config=tensorrt_llm.bindings.executor.KvCacheConfig(
                            free_gpu_memory_fraction=float(
                                os.environ.get("TRT_LLM_MEM_USAGE_FRACTION", "") or 0.4
                            )
                        ),
                    )
        else:
            sys.path.append(os.path.dirname(__file__) + "/VILA")
            import llava
            from llava import conversation as clib

            self._model = llava.load(model_path)
            clib.default_conversation = clib.conv_templates["auto"].copy()
            if os.environ.get("NVILA_VIDEO_MAX_TILES"):
                self._model.config.video_max_tiles = int(os.environ.get("NVILA_VIDEO_MAX_TILES"))
                self._model.llm.config.video_max_tiles = int(
                    os.environ.get("NVILA_VIDEO_MAX_TILES")
                )

        self._output_tpool = concurrent.futures.ThreadPoolExecutor(max_workers=max_batch_size * 2)

        self.model_path = model_path

    @property
    def model_name(self):
        return self._model_name

    def get_conv(self):
        return self._conv.copy()

    def _postprocess(self, output, video_frames_times):
        with TimeMeasure("TRT generate"):
            output.result()
            self._inflight_req_ids.remove(output)
        result = output.outputs[0].text
        if video_frames_times:
            for i, t in enumerate(video_frames_times):
                result = re.sub(f"T{i}(?![0-9])", t, result)
        return [result], [{"input_tokens": 0, "output_tokens": output.outputs[0].length}]

    def can_enqueue_requests(self):
        return len(self._inflight_req_ids) < self._max_batch_size

    def warmup(self):
        from PIL import Image

        self.generate("Say Hi", [Image.new("RGB", (100, 100), color=(255, 255, 255))])

    def generate(self, prompt, images, generation_config=None, video_frames_times=None):
        """Generate a response for prompt using the video embeddings

        Args:
            prompt: Conversation prompt
            video_embeds: Batch of video embeddings
            video_frames_times: Batch of video frame times used for embeddings for each chunk
            generation_config: VLM generation config. Defaults to None.

        Returns:
            List of responses for the batch of chunks
        """

        # Populate default values for the VLM generation parameters
        if not generation_config:
            generation_config = {}

        if "temperature" not in generation_config:
            generation_config["temperature"] = 0.4

        if generation_config["temperature"] == 0:
            generation_config.pop("temperature")

        if "max_new_tokens" not in generation_config:
            generation_config["max_new_tokens"] = 512

        if "top_p" not in generation_config:
            generation_config["top_p"] = 1

        if "top_k" not in generation_config:
            generation_config["top_k"] = 100
        generation_config["top_k"] = int(generation_config["top_k"])

        if "seed" in generation_config:
            seed = generation_config["seed"]
            generation_config.pop("seed")
        else:
            seed = 1

        # Set the seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self._use_trt:

            from tensorrt_llm import SamplingParams

            input = {"prompt": "<vila/video>" + prompt, "multi_modal_data": {"video": [images]}}

            # TRT mode
            output = self._llm.generate_async(
                inputs=input,
                sampling_params=SamplingParams(
                    max_tokens=generation_config.pop("max_new_tokens"),
                    **generation_config,
                    seed=seed,
                ),
            )

            self._inflight_req_ids.append(output)

            return self._output_tpool.submit(
                self._postprocess,
                output,
                video_frames_times,
            )
        else:
            media = {"video": [images]}
            gen_config = self._model.default_generation_config
            gen_config.update(**generation_config)
            result = self._model.generate_content(
                "<vila/video>" + prompt, media=media, generation_config=gen_config
            )
            if video_frames_times:
                for i, t in enumerate(video_frames_times):
                    result = re.sub(f"T{i}(?![0-9])", t, result)
            return [result], [{"input_tokens": 0, "output_tokens": 0}]

    @staticmethod
    def get_model_info():
        return "nvila", "internal", "NVIDIA"


if __name__ == "__main__":
    vila = NVila("/opt/models/llava-v1.5-sharegpt-ptft-13b-lita-soft-ce-im-vid-se", True)
