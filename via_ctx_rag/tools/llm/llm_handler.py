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


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from via_ctx_rag.base import Tool
from via_ctx_rag.utils.globals import DEFAULT_LLM_BASE_URL
from langchain_core.runnables.base import Runnable
from langchain_nvidia_ai_endpoints import register_model, Model, ChatNVIDIA


class LLMTool(Tool, Runnable):
    """A Tool class wrapper for LLMs.

    Returns:
        LLMTool: A Tool that wraps an LLM.
    """

    llm: BaseChatModel

    def __init__(self, llm, name="llm_tool") -> None:
        Tool.__init__(self, name)
        self.llm = llm

    def __getattr__(self, attr):
        return getattr(self.llm, attr)

    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self.llm.stream(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.llm.batch(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.llm.ainvoke(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        return await self.llm.astream(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self.llm.abatch(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.name


class ChatOpenAITool(LLMTool):
    def __init__(
        self, model=None, api_key=None, base_url=DEFAULT_LLM_BASE_URL, **llm_params
    ) -> None:
        if model and model == "gpt-4o":
            base_url = ""
            super().__init__(
                llm=ChatOpenAI(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                )
            )
        elif model and "llama-3.1-70b-instruct" in model and "nvcf" in base_url:
            register_model(
                Model(
                    id=model, model_type="chat", client="ChatNVIDIA", endpoint=base_url
                )
            )
            super().__init__(ChatNVIDIA(model=model, api_key=api_key, **llm_params))
        else:
            super().__init__(
                llm=ChatOpenAI(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                )
            )
