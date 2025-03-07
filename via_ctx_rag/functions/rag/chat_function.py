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

"""function.py: File contains Function class"""

from via_ctx_rag.base import Function


class ChatFunction(Function):
    def setup(self) -> dict:
        self.rag = self.get_param("rag")
        self.chat_config = self._params
        self.graph_db = self.get_tool("graph_db")
        self.chat_llm = self.get_tool("chat_llm")
        self.vector_db = self.get_tool("vector_db")
        self.extraction_function = self.get_function("extraction_function")
        self.retrieval_function = self.get_function("retrieval_function")

        self.graph_post_proccessed = False

    async def acall(self, state: dict) -> dict:
        if (
            self.extraction_function
            and not self.graph_post_proccessed
            and "post_process" in state
        ):
            self.graph_post_proccessed = True
            state = await self.extraction_function.acall(state)
        if "question" in state:
            state = await self.retrieval_function.acall(state)
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        if self.extraction_function:
            await self.extraction_function.aprocess_doc(doc, doc_i, doc_meta)

    async def areset(self, expr):
        self.graph_post_proccessed = False
        if self.extraction_function:
            await self.extraction_function.areset(expr)
        await self.retrieval_function.areset(expr)
