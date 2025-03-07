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
from schema import Schema, Optional

from via_ctx_rag.base import Function
from via_ctx_rag.tools.storage import StorageTool
from via_ctx_rag.utils.ctx_rag_logger import logger, TimeMeasure
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence


class RefineSummarization(Function):
    """Refine Summarization Function"""

    call_schema: Schema = Schema(
        {Optional("start_index"): int, "end_index": int}, ignore_extra_keys=True
    )

    def setup(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_param("prompt")),
                (
                    "user",
                    "Context till now: {previous_summary}.\nNew context: {new_summary}",
                ),
            ]
        )
        self.output_parser = StrOutputParser()
        self.pipeline: RunnableSequence = (
            self.prompt | self.get_tool("llm") | self.output_parser
        )
        self.vector_db: StorageTool = self.get_tool("vector_db")
        self.curr_summary: str = ""
        self.curr_doc_i: int = -1

    async def acall(self, state: dict):
        """refine summarization function call

        Args:
            state (dict): should validate against call_schema
        Returns:
            dict: the state dict will contain result:
            {
                # ...
                # The following key is overwritten or added
                "result" : "summary"
            }
        """
        with TimeMeasure("RefinSumm/Acall", "blue"):
            self.call_schema.validate(state)
            if "start_index" in state and state.get("start_index") != 0:
                raise ValueError("start_index should be 0 for refine summarization")
            if state["end_index"] < 0:
                state["end_index"] = self.curr_doc_i
            result = await self.vector_db.get_text_data(
                fields=["*"], filter=f"doc_i=={state['end_index']}"
            )
            if len(result) == 0:
                state["result"] = ""
                state["error_code"] = "No result found"
            elif len(result) > 0:
                state["result"] = result[0]["text"]
            # Handle more than one result here
            return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        with TimeMeasure("RefineSumm/AprocDoc", "green"):
            logger.debug(f"Refine Summarization add_doc: {doc}, {doc_i}, {doc_meta}")
            self.curr_summary = self.pipeline.invoke(
                {"previous_summary": self.curr_summary, "new_summary": doc}
            )
            self.curr_doc_i = doc_i
            logger.debug(f"Refine summary for {doc_i} is {self.curr_summary}")
            self.vector_db.add_summary(self.curr_summary, metadata={"doc_i": doc_i})
            return self.curr_summary

    async def areset(self, expr=None):
        # TODO: Handle the reset properly. What should be the expr??
        self.vector_db.drop_data(expr)
        await asyncio.sleep(0.001)
        self.curr_summary = ""
