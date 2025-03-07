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

"""context_manager.py: This file implements the context manager.
This module handles managing the input to LLM by calling the handlers of all the
other tools it has access to.
"""

import asyncio
import copy
import traceback
import time
from threading import Thread
from typing import Optional, Dict
import os
import multiprocessing
import concurrent

from via_ctx_rag.base import Function
from via_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from via_ctx_rag.functions.notification import Notifier
from via_ctx_rag.functions.summarization import (
    OfflineBatchSummarization,
    RefineSummarization,
)
from via_ctx_rag.tools.llm import ChatOpenAITool
from via_ctx_rag.tools.notification import AlertSSETool
from via_ctx_rag.tools.storage import MilvusDBTool, Neo4jGraphDB
from via_ctx_rag.utils.globals import (
    DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE,
    DEFAULT_LLM_PARAMS,
    DEFAULT_GRAPH_RAG_BATCH_SIZE,
    DEFAULT_RAG_TOP_K,
)
from via_ctx_rag.functions.rag.chat_function import ChatFunction
from via_ctx_rag.functions.rag.graph_rag.graph_extraction_func import (
    GraphExtractionFunc,
)
from via_ctx_rag.functions.rag.graph_rag.graph_retrieval_func import GraphRetrievalFunc
from via_ctx_rag.functions.rag.vector_rag.vector_retrieval_func import (
    VectorRetrievalFunc,
)

WAIT_ON_PENDING = 10  # Amount of time to wait before clearing the pending

mp_ctx = multiprocessing.get_context("spawn")


class ContextManagerProcess(mp_ctx.Process):
    def __init__(self, config: Dict, req_info):
        logger.info("INITIALIZING CONTEXT MANAGER PROCESS")
        super().__init__()
        self._lock = mp_ctx.Lock()
        self._queue = mp_ctx.Queue()
        self._response_queue = mp_ctx.Queue()
        self._stop = mp_ctx.Event()
        self.auto_indexing: Optional[bool] = None
        self.curr_doc_index: int = -1
        self.config = config
        self.req_info = req_info
        self._pending_add_doc_requests = []

    def _initialize(self):
        self.cm_handler = ContextManagerHandler(self.config, self.req_info)

    def start_bg_loop(self) -> None:
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def start(self):
        self._init_done_event = mp_ctx.Event()
        super().start()

    def stop(self):
        """Stop the process"""
        self._stop.set()
        self.join()

    def run(self) -> None:
        # Run while not signalled to stop
        try:
            self.event_loop = asyncio.new_event_loop()
            self.t = Thread(target=self.start_bg_loop, daemon=True)
            self.t.start()
            self._initialize()

            while not self._stop.is_set():
                with self._lock:
                    qsize = self._queue.qsize()

                    if (qsize) == 0:
                        time.sleep(0.01)
                        continue

                    item = self._queue.get()
                    if item and "add_doc" in item:
                        future = asyncio.run_coroutine_threadsafe(
                            self.cm_handler.aprocess_doc(
                                **item["add_doc"]["doc_content"]
                            ),
                            self.event_loop,
                        )
                        self._pending_add_doc_requests.append(future)
                        future.add_done_callback(self._pending_add_doc_requests.remove)
                    elif item and "reset" in item:
                        expr = item["reset"]
                        with TimeMeasure("context_manager/reset", "green"):
                            stop_time = time.time() + WAIT_ON_PENDING
                            while len(self._pending_add_doc_requests) and (
                                time.time() < stop_time
                            ):
                                time.sleep(2)
                                logger.info(
                                    f"Completing pending requests...{len(self._pending_add_doc_requests)}"
                                )
                                logger.info(f"{self._pending_add_doc_requests}")

                            self.curr_doc_index = -1
                            self.auto_indexing = None  # Reset CM: This ensures next add_doc becomes the first one
                            self._pending_add_doc_requests = []
                            future = asyncio.run_coroutine_threadsafe(
                                self.cm_handler.areset(expr), loop=self.event_loop
                            )
                            future.result()
                    elif item and "call" in item:
                        with TimeMeasure("context_manager/call", "blue"):
                            # TODO: Wait for add docs to finish
                            with TimeMeasure(
                                "context_manager/call/pending_add_doc", "blue"
                            ):
                                concurrent.futures.wait(self._pending_add_doc_requests)

                            state = item["call"]
                            future = asyncio.run_coroutine_threadsafe(
                                self.cm_handler.call(state), self.event_loop
                            )
                            self._response_queue.put(future.result())
                    elif item and "update" in item:
                        self.cm_handler.update(item["update"])

        except Exception as e:
            logger.error("Exception %s", str(e))
            logger.error(traceback.format_exc())

    def add_doc(
        self,
        doc_content: str,
        doc_i: Optional[int] = None,
        doc_meta: Optional[dict] = None,
        callback=None,
    ):
        """
        doc_i can't be < 0
        doc_meta['is_first'] = True is needed to start if not autoindexing
        """
        with TimeMeasure("context_manager/add_doc", "pink"):
            # This is the cm's internal counter of current processing doc.
            # This should start based on is_first.
            # if first doc found then begin processing
            if (
                (not self.auto_indexing)
                and (self.curr_doc_index < 0)
                and doc_meta
                and doc_meta.get("is_first")
            ):
                logger.info(f"First doc found: {doc_i}")
                self.curr_doc_index = doc_i

            if self.auto_indexing:
                doc_i = self.curr_doc_index
            elif doc_i is None:
                raise ValueError("Param doc_i missing.")

            self._queue.put(
                {
                    "add_doc": {
                        "doc_content": {
                            "doc": doc_content,
                            "doc_i": doc_i,
                            "doc_meta": doc_meta,
                        }
                    }
                }
            )

    def update(self, config):
        self._queue.put({"update": config})

    def call(self, state):
        self._queue.put({"call": state})
        return self._response_queue.get()

    def reset(self, expr):
        self._queue.put({"reset": expr})


class ContextManagerHandler:
    """Context Manager: This is the main class that manages the control flow to various handlers.
    This class uses the LLM to determine which handler to call. Prepares the data for the handler
    call. Once the response is ready after multiple handler calls returns the answer to the user.
    """

    # TODO: Is last separately for live stream case
    # TODO: How do we customize prompts from VIA-UI
    # TODO: Runtime Config
    # TODO: Make the functions a list
    # TODO: Unit test for blocking function call when calling another function. Does add_doc block too?
    def __init__(self, config: Dict, req_info) -> None:
        logger.info("INTIALIZING CONTEXT MANAGER HANDLER")

        self._functions: dict[str, Function] = {}
        self.config = config
        self.configure(config, req_info)

    def configure(self, config: Dict, req_info):
        caption_summarization_prompt = ""
        summary_aggregation_prompt = ""

        if req_info:
            caption_summarization_prompt = req_info.caption_summarization_prompt
            summary_aggregation_prompt = req_info.summary_aggregation_prompt

        # cm = ContextManager(config, self._args.log_level)
        summ_config = copy.deepcopy(config.get("summarization"))
        chat_config = copy.deepcopy(config.get("chat"))
        llm_params = summ_config.get("llm", DEFAULT_LLM_PARAMS)
        chat_llm_params = (
            chat_config.get("llm", DEFAULT_LLM_PARAMS)
            if chat_config
            else DEFAULT_LLM_PARAMS
        )
        milvus_db = MilvusDBTool(
            collection_name="summary_till_now_"
            + (req_info.request_id if req_info else "default").replace("-", "_"),
            host=config["milvus_db_host"],
            port=config["milvus_db_port"],
            reranker_base_url=chat_config["reranker"]["base_url"],
            reranker_mode_name=chat_config["reranker"]["model"],
            embedding_base_url=chat_config["embedding"]["base_url"],
            embedding_model_name=chat_config["embedding"]["model"],
        )
        if llm_params["model"] == "gpt-4o":
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            api_key = config["api_key"]
        logger.info("Using %s as the summarization llm", llm_params["model"])
        llm = ChatOpenAITool(api_key=api_key, **llm_params)

        if chat_llm_params["model"] == "gpt-4o":
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            api_key = config["api_key"]
        logger.info("Using %s as the chat llm", chat_llm_params["model"])
        chat_llm = ChatOpenAITool(api_key=api_key, **chat_llm_params)

        try:
            self.default_caption_prompt = summ_config["prompts"]["caption"]
            caption_summarization_prompt = (
                caption_summarization_prompt
                or summ_config["prompts"]["caption_summarization"]
            )
            summ_config["prompts"]["caption_summarization"] = (
                caption_summarization_prompt
            )
            summary_aggregation_prompt = summary_aggregation_prompt or (
                summ_config["prompts"]["summary_aggregation"]
                if "summary_aggregation" in summ_config["prompts"]
                else ""
            )
            summ_config["prompts"]["summary_aggregation"] = summary_aggregation_prompt
        except Exception as e:
            raise ValueError("Prompt(s) missing!") from e

        enable_summarization = True
        if req_info is None or req_info.summarize is None:
            enable_summarization = summ_config["enable"]
        else:
            enable_summarization = req_info.summarize

        if enable_summarization:
            if summ_config["method"] == "batch":
                summ_config["params"] = summ_config.get(
                    "params",
                    {
                        "batch_size": DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE,
                    },
                )
                summ_config["params"]["batch_size"] = summ_config["params"].get(
                    "batch_size", DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE
                )
                try:
                    if req_info and req_info.is_live:
                        logger.debug("Req Info: %s", req_info.summary_duration)
                        logger.debug("Req Info: %s", req_info.chunk_size)
                        summ_config["params"]["batch_size"] = int(
                            req_info.summary_duration / req_info.chunk_size
                        )
                        logger.info(
                            "Overriding batch size to %s for live stream",
                            summ_config["params"]["batch_size"],
                        )
                except Exception as e:
                    logger.error("Overriding batch size failed for live stream: %s", e)
                self.add_function(
                    OfflineBatchSummarization("summarization")
                    .add_tool("llm", llm)
                    .add_tool("vector_db", milvus_db)
                    .config(**summ_config)
                    .done()
                )
            elif summ_config["method"] == "refine":
                self.add_function(
                    RefineSummarization("summarization")
                    .add_tool("llm", llm)
                    .add_tool("vector_db", milvus_db)
                    .config(prompt=caption_summarization_prompt)
                    .done()
                )
            else:
                # should never reach here. Should be validated by the config schema
                raise ValueError("Incorrect summarization config")
        else:
            logger.info("Summarization disabled with the API call")

        if req_info is None or req_info.enable_chat:
            if chat_config["rag"] != "vector-rag" and chat_config["rag"] != "graph-rag":
                logger.info(
                    "Both graph_rag and vector_rag are disabled. Q&A is disabled"
                )
            else:
                logger.info("Setting up QnA, rag type: %s", chat_config["rag"])
                neo4j_uri = os.getenv("GRAPH_DB_URI")
                if not neo4j_uri:
                    raise ValueError("GRAPH_DB_URI not set. Please set GRAPH_DB_URI.")
                neo4j_username = os.getenv("GRAPH_DB_USERNAME")
                if not neo4j_username:
                    raise ValueError(
                        "GRAPH_DB_USERNAME not set. Please set GRAPH_DB_USERNAME."
                    )
                neo4j_password = os.getenv("GRAPH_DB_PASSWORD")
                if not neo4j_password:
                    raise ValueError(
                        "GRAPH_DB_PASSWORD not set. Please set GRAPH_DB_PASSWORD."
                    )
                chat_config["params"] = chat_config.get(
                    "params",
                    {
                        "batch_size": DEFAULT_GRAPH_RAG_BATCH_SIZE,
                        "top_k": DEFAULT_RAG_TOP_K,
                    },
                )
                chat_config["params"]["batch_size"] = chat_config["params"].get(
                    "batch_size", DEFAULT_GRAPH_RAG_BATCH_SIZE
                )
                chat_config["params"]["top_k"] = chat_config["params"].get(
                    "top_k", DEFAULT_RAG_TOP_K
                )
                if chat_config["rag"] == "graph-rag":
                    neo4jDB = Neo4jGraphDB(
                        url=neo4j_uri,
                        username=neo4j_username,
                        password=neo4j_password,
                        embedding_model_name=chat_config["embedding"]["model"],
                        embedding_base_url=chat_config["embedding"]["base_url"],
                    )

                    self.add_function(
                        ChatFunction("chat")
                        .add_function(
                            "extraction_function",
                            GraphExtractionFunc("extraction_function")
                            .add_tool("graph_db", neo4jDB)
                            .add_tool("chat_llm", chat_llm)
                            .config(**chat_config)
                            .done(),
                        )
                        .add_function(
                            "retrieval_function",
                            GraphRetrievalFunc("retrieval_function")
                            .add_tool("graph_db", neo4jDB)
                            .add_tool("chat_llm", chat_llm)
                            .config(**chat_config)
                            .done(),
                        )
                        .config(**chat_config)
                        .done(),
                    )
                elif chat_config["rag"] == "vector-rag":
                    self.add_function(
                        ChatFunction("chat")
                        .add_function(
                            "retrieval_function",
                            VectorRetrievalFunc("retrieval_function")
                            .add_tool("vector_db", milvus_db)
                            .add_tool("chat_llm", chat_llm)
                            .config(**chat_config)
                            .done(),
                        )
                        .config(**chat_config)
                        .done(),
                    )
        else:
            if req_info and req_info.enable_chat is False:
                logger.info("Chat/Q&A disabled with the API call")

        notification_config = config.get("notification")
        if notification_config and notification_config.get("enable"):
            notification_llm_params = notification_config.get("llm")
            notification_params = notification_config.get("params", {})
            if notification_llm_params["model"] == "gpt-4o":
                api_key = os.environ["OPENAI_API_KEY"]
            else:
                api_key = config["api_key"]
            logger.info(
                "Using %s as the notification llm", notification_llm_params["model"]
            )
            notification_llm = ChatOpenAITool(
                api_key=api_key, **notification_llm_params
            )
            self.add_function(
                Notifier("notification")
                .add_tool("llm", notification_llm)
                .add_tool(
                    "notification_tool",
                    AlertSSETool(endpoint=notification_config.get("endpoint")),
                )
                .config(**notification_params)
                .done()
            )
        else:
            logger.info("Notifications disabled")

    def add_function(self, f: Function):
        assert f.name not in self._functions, str(self._functions)
        self._functions[f.name] = f
        return self

    def get_function(self, fname):
        return self._functions[fname] if fname in self._functions else None

    def update(self, config):
        logger.info("Updating context manager with config:\n%s", config)
        try:
            for fn, fn_config in config.items():
                if fn in self._functions:
                    self._functions[fn].update(**fn_config)
        except Exception as e:
            logger.error("Overriding failed for config %s with error: %s", config, e)
            logger.error(traceback.format_exc())

    async def aprocess_doc(self, doc, doc_i, doc_meta):
        tasks = []
        with TimeMeasure("context_manager/aprocess_doc", "yellow"):
            for _, f in self._functions.items():
                tasks.append(
                    asyncio.create_task(
                        f.aprocess_doc_(doc, doc_i, doc_meta), name=f.name
                    )
                )
            return await asyncio.gather(*tasks)

    async def call(self, state):
        results = {}
        with TimeMeasure("context_manager/call", "green"):
            tasks = []
            task_results = []
            for func, call_params in state.items():
                tasks.append(
                    asyncio.create_task(self._functions[func](call_params), name=func)
                )
            task_results = await asyncio.gather(*tasks)
            for index, func in enumerate(state):
                results[func] = task_results[index]
        return results

    async def areset(self, expr):
        tasks = []
        for _, f in self._functions.items():
            tasks.append(asyncio.create_task(f.areset(expr), name=f.name))
        return await asyncio.gather(*tasks)


class ReqInfo:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class ContextManager:
    def __init__(self, config: Dict, req_info, log_level="info") -> None:
        logger.info("INTIALIZING CONTEXT MANAGER")
        logger.setLevel(log_level.upper())
        req_info_obj = None
        if req_info:
            req_info_obj = ReqInfo(
                **{
                    "summarize": req_info.summarize,
                    "enable_chat": req_info.enable_chat,
                    "is_live": req_info.is_live,
                    "request_id": req_info.request_id,
                    "caption_summarization_prompt": req_info.caption_summarization_prompt,
                    "summary_aggregation_prompt": req_info.summary_aggregation_prompt,
                    "chunk_size": req_info.chunk_size,
                    "summary_duration": req_info.summary_duration,
                }
            )

        self.process = ContextManagerProcess(config, req_info_obj)
        self.process.start()

    def __del__(self):
        self.process.stop()

    def add_doc(
        self,
        doc_content: str,
        doc_i: Optional[int] = None,
        doc_meta: Optional[dict] = None,
        callback=None,
    ):
        """
        Thread-safe method to add a document.
        """
        self.process.add_doc(doc_content, doc_i, doc_meta, callback)

    def update(self, config):
        self.process.update(config=config)

    def call(self, state):
        return self.process.call(state)

    def reset(self, expr):
        return self.process.reset(expr)
