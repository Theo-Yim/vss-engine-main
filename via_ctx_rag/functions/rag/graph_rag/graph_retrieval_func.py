################################################################################
# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

import asyncio
import os
from pathlib import Path
from re import compile
import traceback

from langchain_core.output_parsers import StrOutputParser

from via_ctx_rag.base import Function
from via_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from via_ctx_rag.tools.health.rag_health import GraphMetrics
from via_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from via_ctx_rag.functions.rag.graph_rag.graph_retrieval import GraphRetrieval
from via_ctx_rag.utils.globals import DEFAULT_RAG_TOP_K
from langchain_core.messages import HumanMessage, AIMessage
from via_ctx_rag.utils.utils import remove_think_tags


class GraphRetrievalFunc(Function):
    """GraphRetrievalFunc Function"""

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
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.graph_retrieval = GraphRetrieval(
            llm=self.chat_llm, graph=self.graph_db, top_k=self.top_k
        )
        self.regex_object = compile(r"<(\d+[.]\d+)>")

    async def acall(self, state: dict) -> dict:
        if self.log_dir:
            with TimeMeasure("GraphRAG/aprocess-doc/metrics_dump", "yellow"):
                log_path = Path(self.log_dir).joinpath("graph_rag_metrics.json")
                self.metrics.dump_json(log_path.absolute())
        try:
            question = state.get("question", "").strip()
            if not question:
                raise ValueError("No input provided in state.")

            if question.lower() == "/clear":
                logger.debug("Clearing chat history...")
                self.graph_retrieval.clear_chat_history()
                state["response"] = "Cleared chat history"
                return state

            with TimeMeasure("GraphRetrieval/HumanMessage", "blue"):
                user_message = HumanMessage(content=question)
                self.graph_retrieval.add_message(user_message)

            docs = self.graph_retrieval.retrieve_documents()

            if docs:
                formatted_docs = self.graph_retrieval.process_documents(docs)
                ai_response = self.graph_retrieval.get_response(
                    question, formatted_docs
                )
                answer = remove_think_tags(ai_response.content)
            else:
                formatted_docs = "No documents retrieved."
                answer = "No answer could be generated due to lack of documents."

            with TimeMeasure("GraphRetrieval/AIMsg", "red"):
                ai_message = AIMessage(content=answer)
                self.graph_retrieval.add_message(ai_message)

            self.graph_retrieval.summarize_chat_history()

            logger.debug("Summarizing chat history thread started.")

            state["response"] = answer
            state["response"] = self.regex_object.sub(r"\g<1>", state["response"])

            if "formatted_docs" in state:
                state["formatted_docs"].append(formatted_docs)
            else:
                state["formatted_docs"] = [formatted_docs]

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in QA %s", str(e))
            state["response"] = "That didn't work. Try another question."

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        pass

    async def areset(self, expr):
        self.graph_retrieval.clear_chat_history()
        await asyncio.sleep(0.01)
