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

import asyncio
import os
import traceback
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from via_ctx_rag.base import Function
from via_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from via_ctx_rag.tools.health.rag_health import GraphMetrics
from via_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from via_ctx_rag.functions.rag.graph_rag.graph_extraction import GraphExtraction
from via_ctx_rag.utils.ctx_rag_batcher import Batcher
from via_ctx_rag.utils.globals import DEFAULT_RAG_TOP_K


class GraphExtractionFunc(Function):
    """GraphExtractionFunc Function"""

    config: dict
    output_parser = StrOutputParser()
    graph_db: Neo4jGraphDB
    metrics = GraphMetrics()
    if os.environ.get("VSS_LOG_LEVEL") == "":
        logger.setLevel("INFO")
    else:
        logger.setLevel(os.environ.get("VSS_LOG_LEVEL", "INFO").upper())

    def setup(self):
        self.graph_db = self.get_tool("graph_db")
        self.chat_llm = self.get_tool("chat_llm")
        self.rag = self.get_param("rag")
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.batch_size = self.get_param("params", "batch_size")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.batcher = Batcher(self.batch_size)
        self.graph_extraction = GraphExtraction(
            batcher=self.batcher, llm=self.chat_llm, graph=self.graph_db
        )
        self.graph_create_start = None

    async def acall(self, state: dict):
        await self.graph_extraction.apost_process()
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        """QnA process doc call"""
        with TimeMeasure("GraphRAG/aprocess-doc:", "blue") as tm:
            if not doc_meta["is_last"]:
                if doc_meta["file"].startswith("rtsp://"):
                    # if live stream summarization
                    doc = f"<{doc_meta['start_ntp']}> <{doc_meta['end_ntp']}> " + doc
                else:
                    # if file summmarization
                    doc = (
                        f"<{doc_meta['start_pts'] / 1e9:.2f}> <{doc_meta['end_pts'] / 1e9:.2f}> "
                        + doc
                    )
            batch = self.batcher.add_doc(doc, doc_i=doc_i, doc_meta=doc_meta)
            if batch.is_full():
                with TimeMeasure(
                    "GraphRAG/aprocess-doc/graph-create: "
                    + str(self.batcher.get_batch_index(doc_i)),
                    "green",
                ):
                    try:
                        with get_openai_callback() as cb:
                            await self.graph_extraction.acreate_graph(batch)
                        logger.info(
                            "GraphRAG Creation for %d docs\n"
                            "Total Tokens: %s, "
                            "Prompt Tokens: %s, "
                            "Completion Tokens: %s, "
                            "Successful Requests: %s, "
                            "Total Cost (USD): $%s"
                            % (
                                batch._batch_size,
                                cb.total_tokens,
                                cb.prompt_tokens,
                                cb.completion_tokens,
                                cb.successful_requests,
                                cb.total_cost,
                            ),
                        )
                        self.metrics.graph_create_tokens += cb.total_tokens
                        self.metrics.graph_create_requests += cb.successful_requests
                    except Exception as e:
                        logger.error(traceback.format_exc())
                        logger.error(
                            "GraphRAG/aprocess-doc Failed with error %s\n Skipping...",
                            e,
                        )
                        return "Failed"
        if self.graph_create_start is None:
            self.graph_create_start = tm.start_time
        self.metrics.graph_create_latency = tm.end_time - self.graph_create_start
        return "Success"

    async def areset(self, expr):
        self.batcher.flush()
        self.graph_create_start = None
        self.graph_extraction.cleaned_graph_documents_list = []
        self.metrics.reset()

        self.graph_db.run_cypher_query("MATCH (n) DETACH DELETE n")

        await asyncio.sleep(0.01)
