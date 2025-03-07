################################################################################
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

"""summarization.py: File contains Function class"""

import asyncio
import math
import time
from schema import Schema

from via_ctx_rag.base import Function
from via_ctx_rag.tools.storage import StorageTool
from via_ctx_rag.utils.ctx_rag_logger import logger, TimeMeasure

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence


class BatchSummarization(Function):
    """Refine Summarization Function"""

    config: dict
    batch_prompt: str
    aggregation_prompt: str
    output_parser = StrOutputParser()
    batch_size: int
    curr_batch: str
    curr_summary: str = ""
    curr_batch_size: int
    curr_batch_i: int
    batch_pipeline: RunnableSequence
    aggregation_pipeline: RunnableSequence
    vector_db: StorageTool
    curr_batch_i: int
    timeout: int = 30  # seconds
    call_schema: Schema = Schema(
        {"start_index": int, "end_index": int}, ignore_extra_keys=True
    )

    def setup(self):
        # fixed params
        self.batch_prompt = ChatPromptTemplate.from_messages(
            [("system", self.get_param("summarization_prompt")), ("user", "{input}")]
        )
        self.aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_param("summary_aggregation_prompt")),
                ("user", "{input}"),
            ]
        )
        self.output_parser = StrOutputParser()
        self.batch_pipeline = (
            self.batch_prompt | self.get_tool("llm") | self.output_parser
        )
        self.aggregation_pipeline = (
            self.aggregation_prompt | self.get_tool("llm") | self.output_parser
        )
        self.batch_size = self.get_param("batch_size")
        self.vector_db = self.get_tool("vector_db")
        self.timeout = (
            self.get_param("timeout_sec", required=False)
            if self.get_param("timeout_sec", required=False)
            else self.timeout
        )

        # working params
        self.curr_summary = ""
        self.curr_batch = ""
        self.curr_batch_i = 0
        self.curr_batch_size = 0

    async def acall(self, state: dict):
        """batch summarization function call

        Args:
            state (dict): should validate against call_schema
        Returns:
            dict: the state dict will contain result:
            {
                # ...
                # The following key is overwritten or added
                "result" : "summary",
                "error_code": "Error String" # Optional
            }
        """
        with TimeMeasure("BatchSumm/Acall", "green"):
            result = []
            self.call_schema.validate(state)
            stop_time = time.time() + self.timeout
            target_start_batch_index = int(
                math.ceil(state["start_index"] / self.batch_size)
            )
            target_end_batch_index = int(
                math.floor(state["end_index"] / self.batch_size)
            )
            if target_end_batch_index == -1:
                logger.debug(f"Current batch index: {self.curr_batch_i}")
                target_end_batch_index = self.curr_batch_i
            while time.time() < stop_time:
                result = await self.vector_db.get_text_data(
                    fields=["text"],
                    filter=f"{target_start_batch_index}<=batch_i<={target_end_batch_index}",
                )
                logger.debug(f"Target Batch Start: {target_start_batch_index}")
                logger.debug(f"Target Batch End: {target_end_batch_index}")
                logger.info(
                    f"Waiting for {target_end_batch_index-target_start_batch_index+1}..."
                )
                logger.debug(f"Length of Results: {len(result)}")
                # logger.debug(
                #     f"Data between {target_start_batch_index} and {target_end_batch_index}: {result}")
                if len(result) == target_end_batch_index - target_start_batch_index + 1:
                    logger.info(f"Length of Results: {len(result)}")
                    break
            if len(result) == 0:
                state["result"] = ""
                state["error_code"] = "No batch summaries found"
                logger.error("No batch summaries found")
            elif len(result) > 0:
                combined_batch = ""
                for r in result:
                    combined_batch += r["text"]
                state["result"] = await self.aggregation_pipeline.ainvoke(
                    combined_batch
                )
            return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        with TimeMeasure("BatchSumm/AprocDoc", "red"):
            self.curr_batch += "\n" + doc
            self.curr_batch_size = self.curr_batch_size + 1
            if self.curr_batch_size == self.batch_size or (
                doc_meta and "is_last" in doc_meta and doc_meta["is_last"]
            ):
                doc_meta["batch_i"] = self.curr_batch_i
                self.curr_summary = await self.batch_pipeline.ainvoke(self.curr_batch)
                self.vector_db.add_summary(self.curr_summary, metadata=doc_meta)
                self.curr_batch_i += 1
                self.curr_batch = ""
                self.curr_batch_size = 0
            return self.curr_summary

    async def areset(self, expr=None):
        # TODO: use async method for drop data
        await asyncio.sleep(0.001)
        self.vector_db.drop_data(expr)
        self.curr_summary = ""
        self.curr_batch = ""
        self.curr_batch_i = 0
        self.curr_batch_size = 0
