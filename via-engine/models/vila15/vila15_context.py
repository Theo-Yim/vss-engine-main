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

import os
import sys

import torch

from chunk_info import ChunkInfo

sys.path.append(os.path.dirname(__file__) + "/VILA")


class Vila15Context:
    """VILA 1.5 VLM Model Conversation context.

    This helps maintain conversation context for different chunks/files while
    not having to reload the actual model."""

    def __init__(self, model) -> None:
        """Vila15Context constructor

        Args:
            model: Vila15Model instance
        """
        # Get the conversation object
        self._conv = model.get_conv()
        self._model = model
        if "mpt" in model.model_name.lower():
            self._roles = ("user", "assistant")
        else:
            self._roles = self._conv.roles
        self.messages = []

    def set_video_embeds(self, chunks, video_embeds, video_frames, video_frames_times):
        """Set the chunks, and corresponding video embeddings and frame times.
        Accepts batched inputs (lists)"""
        self._video_frames_times = video_frames_times
        self._chunks = chunks

        self._video_embeds = video_embeds
        self._video_embeds = torch.stack([v.half().cuda() for v in self._video_embeds])

    def ask(self, query, respond=True, skip_time_tokens=False, generation_config=None, chunk=None):
        """Ask a query to the model

        Args:
            query: Prompt for the VLM model
            respond: If true, generate response. If false, only add to the conversation context.
                     Defaults to True.
            skip_time_tokens: Skip decoding time tokens in the response. Defaults to False.
            generation_config: Dictionary of VLM output parameters (top-k, seed etc).
                               Defaults to None.

        Returns:
            List of VLM responses per chunk for the batched input
        """

        if "<image>" in query:
            self._conv.messages = []

        # Add the <image> tag to the prompt, to mark where the video embeddings should be inserted
        inp = "<image>\n" * len(self._video_frames_times[0])

        # Add the user prompt to the conversation context
        self._conv.append_message(self._conv.roles[0], inp + query)
        self._conv.append_message(self._conv.roles[1], None)

        if not respond:
            # Only for adding the prompt to the context. No need for VLM response
            return

        # Convert the conversation to a string prompt
        prompt = self._conv.get_prompt()

        # Generate a response from the VLM model
        return self._model.generate(
            prompt, self._video_embeds, self._video_frames_times, generation_config, chunk=chunk
        )


if __name__ == "__main__":

    import argparse

    from vlm_pipeline.video_file_frame_getter import (
        DefaultFrameSelector,
        VideoFileFrameGetter,
    )

    from .vila15_embedding_generator import Vila15EmbeddingGenerator
    from .vila15_model import Vila15

    parser = argparse.ArgumentParser(description="VILA Context")
    parser.add_argument("file", type=str, help="File to run VILA on")
    parser.add_argument("query", type=str, help="Query to run on the video")

    parser.add_argument(
        "--start-time", type=int, default=0, help="Start time in sec to get frames from"
    )

    parser.add_argument(
        "--end-time", type=int, default=-1, help="End time in sec to get frames from"
    )
    parser.add_argument("--use-trt", action="store_true", help="End time in sec to get frames from")

    args = parser.parse_args()

    lita = Vila15("/opt/models/llava-v1.5-sharegpt-ptft-13b-lita-soft-ce-im-vid-se", args.use_trt)
    lita_ctx = Vila15Context(lita)

    chunk = ChunkInfo()
    chunk.file = args.file
    chunk.start_pts = args.start_time * 1000000000
    chunk.end_pts = args.end_time * 1000000000 if args.end_time >= 0 else -1

    frame_getter = VideoFileFrameGetter(DefaultFrameSelector(100))
    frames, frames_pts = frame_getter.get_frames(chunk)
    embeds = Vila15EmbeddingGenerator(
        "/opt/models/llava-v1.5-sharegpt-ptft-13b-lita-soft-ce-im-vid-se", False
    ).get_embeddings(
        [frames]
    )  # args.use_trt

    lita_ctx.set_video_embeds([chunk], embeds, None, [frames_pts])
    print(lita_ctx.ask(args.query)[0])
