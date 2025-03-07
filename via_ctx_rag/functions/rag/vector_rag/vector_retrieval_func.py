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

"""graph_rag.py: File contains Function class"""

import os
from pathlib import Path
from re import compile
import traceback
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

from via_ctx_rag.base import Function
from via_ctx_rag.tools.storage.milvus_db import MilvusDBTool
from via_ctx_rag.tools.health.rag_health import GraphMetrics
from via_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from via_ctx_rag.utils.globals import DEFAULT_RAG_TOP_K


class VectorRetrievalFunc(Function):
    """VectorRAG Function"""

    config: dict
    output_parser = StrOutputParser()
    vector_db: MilvusDBTool
    metrics = GraphMetrics()
    if os.environ.get("VSS_LOG_LEVEL") == "":
        logger.setLevel("INFO")
    else:
        logger.setLevel(os.environ.get("VSS_LOG_LEVEL", "INFO").upper())

    def setup(self):
        self.chat_llm = self.get_tool("chat_llm")
        self.vector_db = self.get_tool("vector_db")
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.regex_object = compile(r"<(\d+[.]\d+)>")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.vector_db.reranker,
            base_retriever=self.vector_db.vector_db.as_retriever(
                search_kwargs={"filter": {"doc_type": "caption"}, "k": self.top_k}
            ),
        )
        self.g_semantic_sim_chain = RetrievalQA.from_chain_type(
            llm=self.chat_llm, retriever=self.compression_retriever
        )

    async def acall(self, state: dict):
        """QnA function call"""
        if self.log_dir:
            with TimeMeasure("GraphRAG/aprocess-doc/metrics_dump", "yellow"):
                log_path = Path(self.log_dir).joinpath("vector_rag_metrics.json")
                self.metrics.dump_json(log_path.absolute())
        try:
            logger.debug("Running qna with question: %s", state["question"])
            with TimeMeasure("VectorRAG/retrieval", "red"):
                semantic_search_answer = await self.g_semantic_sim_chain.ainvoke(
                    state["question"]
                )
                logger.debug(
                    "Semantic search response: %s", semantic_search_answer["result"]
                )
                state["response"] = semantic_search_answer["result"]
                state["response"] = self.regex_object.sub(r"\g<1>", state["response"])
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in QA %s", str(e))
            state["response"] = "That didn't work. Try another question."
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        pass
