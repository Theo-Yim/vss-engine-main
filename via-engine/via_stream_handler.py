################################################################################
# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################
"""VIA Pipeline

VIA Pipeline goes through the following initialization steps:
1. Initialize some internal structures required for managing the VIA pipeline
    - A map (_request_info_map) of RequestInfo - summarization request and response details
    - A map (_live_stream_info_map) of LiveStreamInfo - Live stream details
    - EmbeddingHelper (_emb_helper) - to save/retrieve video embeddings

2. Download model from NGC (Only if model-path is set to NGC model)
    - This step is skipped if a non VILA VLM model is used or local model path
      is provided or the model is found in NGC model cache
3. Build TRT engine for the model (Only for VILA model)
    - This step is skipped for non VILA VLM models or if TRT engine is found in
      cache.
4. Start processes per GPU:
    a. Decoder Process (DecoderProcess)
    b. Embedding Generation Process (EmbeddingProcess)
    c. VLM Process (VlmProcess)

5. Initialize Guardrails (_rails)
6. Initialize milvus DB connection (_summary_db_handler) and CA-RAG (_ctx_mgr)
7. Wait for all processes started in (3) to complete initialization.
8. Start a thread (_processed_chunk_queue_watcher_thread) to gather and aggregate
   chunks processed by VLM pipeline and run post-processing steps like summarization
   using CA-RAG.

VLM pipeline looks like below

- Once a summary request is triggered for a file / live stream, the splitter will generate chunks
  and distribute them across the N decoder processes.
- The decoder processes will decoding chunks and picking N frames from it, preprocess the frames
  and push it to the embedding generation process
- The embedding generation process generates video embeddings per chunk from the pre-processed
  frames. Depending on the model used this step might be skipped. The embeddings are pushed to
  VLM process.
- The VLM process is responsible for executing the VLM model. It generates a response from the
  VLM model with a prompt and video embeddings as input. It pushes this summary to the output
  watcher thread.
- Watcher thread gathers all chunk responses. Once all chunks are processed for a file or summary
  duration is reached for a live stream, it calls the CA-RAG to aggregate and summarize the
  chunk responses and generate the final response to be sent to the user.
"""

from vlm_pipeline import VlmPipeline, VlmRequestParams, VlmChunkResponse  # isort:skip
import argparse
import asyncio
import concurrent.futures
import copy
import json
import logging
import os
import shutil
import sys
import textwrap
import time
import traceback
import uuid
from argparse import ArgumentParser
from datetime import datetime, timezone
from enum import Enum
from threading import RLock, Thread

import aiohttp
import cuda
import cuda.cudart
import gi
import jinja2
import nvtx
import prometheus_client as prom
import uvicorn
import yaml
from fastapi import FastAPI
from tabulate import tabulate
from via_ctx_rag.context_manager import ContextManager

from asset_manager import Asset
from chunk_info import ChunkInfo
from utils import MediaFileInfo
from via_exception import ViaException
from via_health_eval import GPUMonitor, RequestHealthMetrics
from via_logger import TimeMeasure, logger

DEFAULT_CALLBACK_JSON_TEMPLATE = (
    "{ "
    '"streamId": "{{ streamId }}", '
    '"alertId": "{{ alertId }}", '
    '"ntpTimestamp": "{{ ntpTimestamp }}", '
    '"alertDetails": "{{ alertText }}", '
    '"detectedEvents": {{ detectedEvents }}'
    "}"
)

ALERT_CALLBACK_PORT = 60000


class AlertInfo:
    """Store information for an alert"""

    def __init__(self):
        self.alert_id = str(uuid.uuid4())
        self.events: list[str] = []
        self.callbackUrl = ""
        self.callbackJsonTemplate = DEFAULT_CALLBACK_JSON_TEMPLATE
        self.callbackToken = ""
        self.liveStreamId = ""
        self.requestId = ""
        self.alert_tool: AlertSseTool | AlertCallbackTool = None


class RequestInfo:
    """Store information for a request"""

    class Status(Enum):
        """Video Query Request Status."""

        QUEUED = "queued"
        PROCESSING = "processing"
        SUCCESSFUL = "successful"
        FAILED = "failed"
        STOPPING = "stopping"

    class Response:
        def __init__(self, start_timestamp: str, end_timestamp: str, response: str) -> None:
            self.start_timestamp = start_timestamp
            self.end_timestamp = end_timestamp
            self.response = response

    class Alert:
        offset = 0
        ntpTimestamp = ""
        detectedEvents: list[str] = []
        streamId = ""
        name = ""
        details = ""

    def __init__(self) -> None:
        self.request_id = str(uuid.uuid4())
        self.chunk_count = 0
        self.chunk_size = 0
        self.chunk_overlap_duration = 0
        self.file = ""
        self.processed_chunk_list: list[VlmChunkResponse] = []
        self.is_summarization = False
        self.vlm_request_params = VlmRequestParams()
        self.progress = 0
        self.response: list[RequestInfo.Response] = []
        self.is_live = False
        self.start_timestamp = None
        self.end_timestamp = None
        self.queue_time = None
        self.start_time = None
        self.end_time = None
        self.file_duration = 0
        self.assets: list[Asset] = None
        self.status = RequestInfo.Status.QUEUED
        self.summary_duration = 0
        self.caption_summarization_prompt = ""
        self.summary_aggregation_prompt = ""
        self.graph_rag_prompt_yaml = ""
        self._health_summary = None
        self._monitor = None
        self._ca_rag_latency = 0
        self._ctx_mgr: ContextManager = None
        self._output_process_thread_pool: concurrent.futures.ThreadPoolExecutor = None
        self.alerts: list[RequestInfo.Alert] = []
        self.nvtx_vlm_start = None
        self.nvtx_summarization_start = None
        self.summarize = None
        self.enable_chat = True
        self.enable_cv_pipeline = False
        self.cv_metadata_json_file = ""
        self.pending_add_doc_start_time = 0
        self.pending_add_doc_end_time = 0
        self.num_frames_per_chunk = None
        self.summarize_batch_size = None
        self.rag_batch_size = None
        self.rag_type = None
        self.rag_top_k = None
        self.vlm_input_width = None
        self.vlm_input_height = None
        self.last_chunk: ChunkInfo | None = None


class DCSerializer:
    @staticmethod
    def to_json(request_info: RequestInfo, file_path):
        try:
            with open(file_path, "w") as f:
                for vlm_response in request_info.processed_chunk_list:
                    json.dump(
                        {
                            "vlm_response": vlm_response.vlm_response,
                            "frame_times": vlm_response.frame_times,
                            "chunk": {
                                "streamId": vlm_response.chunk.streamId,
                                "chunkIdx": vlm_response.chunk.chunkIdx,
                                "file": vlm_response.chunk.file,
                                "pts_offset_ns": vlm_response.chunk.pts_offset_ns,
                                "start_pts": vlm_response.chunk.start_pts,
                                "end_pts": vlm_response.chunk.end_pts,
                                "start_ntp": vlm_response.chunk.start_ntp,
                                "end_ntp": vlm_response.chunk.end_ntp,
                                "start_ntp_float": vlm_response.chunk.start_ntp_float,
                                "end_ntp_float": vlm_response.chunk.end_ntp_float,
                                "is_first": vlm_response.chunk.is_first,
                                "is_last": vlm_response.chunk.is_last,
                            },
                        },
                        f,
                    )
                    f.write("\n")
        except Exception as e:
            logger.warning("write to_json Exception:", str(e))

    @staticmethod
    def from_json(file_path):
        request_info = RequestInfo()
        try:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    chunk_info = ChunkInfo()
                    chunk_info.streamId = data["chunk"]["streamId"]
                    chunk_info.chunkIdx = data["chunk"]["chunkIdx"]
                    chunk_info.file = data["chunk"]["file"]
                    chunk_info.pts_offset_ns = data["chunk"]["pts_offset_ns"]
                    chunk_info.start_pts = data["chunk"]["start_pts"]
                    chunk_info.end_pts = data["chunk"]["end_pts"]
                    chunk_info.start_ntp = data["chunk"]["start_ntp"]
                    chunk_info.end_ntp = data["chunk"]["end_ntp"]
                    chunk_info.start_ntp_float = data["chunk"]["start_ntp_float"]
                    chunk_info.end_ntp_float = data["chunk"]["end_ntp_float"]
                    chunk_info.is_first = data["chunk"]["is_first"]
                    chunk_info.is_last = data["chunk"]["is_last"]

                    vlm_response = VlmChunkResponse()
                    vlm_response.vlm_response = data["vlm_response"]
                    vlm_response.frame_times = data["frame_times"]
                    vlm_response.chunk = chunk_info

                    request_info.processed_chunk_list.append(vlm_response)
                # Sort the processed_chunk_list by chunkIdx
                if request_info.processed_chunk_list:
                    request_info.processed_chunk_list.sort(key=lambda x: x.chunk.chunkIdx)
        except Exception as e:
            logger.warning("read from json exception", str(e))
        return request_info


class LiveStreamInfo:
    """Store information for a live stream"""

    def __init__(self) -> None:
        self.chunk_size = 0
        self.req_info: list[RequestInfo] = []
        self.asset: Asset = None
        self.stop = False
        self.live_stream_ended = False
        self.pending_futures = []


def ntp_to_unix_timestamp(ntp_ts):
    """Convert an RFC3339 timestamp string to a UNIX timestamp(float)"""
    return (
        datetime.strptime(ntp_ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
    )


class AlertCallbackTool:
    def __init__(self, name, alert_info: AlertInfo):
        self.name = name
        self._alert_info = alert_info

    async def notify(self, title: str, message: str, metadata: dict):
        try:
            doc = metadata["doc"]
            events_detected = metadata["events_detected"]
            callback_json = jinja2.Template(self._alert_info.callbackJsonTemplate).render(
                streamId=self._alert_info.liveStreamId,
                alertId=self._alert_info.alert_id,
                ntpTimestamp=metadata["start_ntp"],
                alertText=json.dumps(doc)[1:-1],
                detectedEvents=json.dumps(events_detected),
            )
            headers = (
                {"Authorization": f"Bearer {self._alert_info.callbackToken}"}
                if self._alert_info.callbackToken
                else {}
            )
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._alert_info.callbackUrl, json=json.loads(callback_json), headers=headers
                ) as r:
                    r.raise_for_status()
        except Exception as ex:
            logger.error(
                "Alert callback failed for event(s) '%s' - %s", ", ".join(events_detected), str(ex)
            )


class AlertSseTool:
    def __init__(self, name, sse_tool_name, req_info: RequestInfo):
        self.name = name
        self._req_info = req_info
        self._sse_tool_name = sse_tool_name

    async def notify(self, title: str, message: str, metadata: dict):
        alert = RequestInfo.Alert()
        alert.details = metadata["doc"]
        alert.detectedEvents = metadata["events_detected"]
        alert.name = self._sse_tool_name
        if self._req_info.is_live:
            alert.ntpTimestamp = metadata["start_ntp"]
        else:
            alert.offset = int(metadata["start_pts"] / 1e9)
        alert.streamId = metadata["streamId"]
        self._req_info.alerts.append(alert)


class ViaStreamHandler:
    """VIA Stream Handler"""

    class Metrics:
        def __init__(self) -> None:
            """Initialize the VIA Stream Handler metrics.
            Metrics are based on the prometheus client."""
            self.queries_processed = prom.Counter(
                "video_file_queries_processed",
                "Number of video file queries whose processing is complete",
            )
            self.queries_pending = prom.Gauge(
                "video_file_queries_pending",
                "Number of video file queries which are queued and yet to be processed",
            )

            self.active_live_streams = prom.Gauge(
                "active_live_streams",
                "Number of live streams whose summaries are being actively generated",
            )

        def unregister(self):
            prom.REGISTRY.unregister(self.queries_processed)
            prom.REGISTRY.unregister(self.queries_pending)
            prom.REGISTRY.unregister(self.active_live_streams)

    def __init__(self, args) -> None:
        """Initialize the VIA Stream Handler"""
        logger.info("Initializing VIA Stream Handler")

        self._notification_llm_api_key = None
        self._notification_llm_params = None

        self._start_time = time.time()

        self._metrics = ViaStreamHandler.Metrics()

        self._lock = RLock()
        self._request_info_map: dict[str, RequestInfo] = {}
        self._live_stream_info_map: dict[str, LiveStreamInfo] = {}
        self._alert_info_map: dict[str, AlertInfo] = {}
        self._args = args

        self._vlm_pipeline = VlmPipeline(args.asset_dir, args)
        self._via_health_eval = False

        self._start_ca_rag_alert_handler()

        self.default_caption_prompt = self._args.summarization_query

        if not args.disable_ca_rag:
            try:
                try:
                    with open(args.ca_rag_config, mode="r", encoding="utf8") as c:
                        config = yaml.safe_load(c)
                except Exception as e:
                    self.stop(True)
                    raise ValueError(f"{args.ca_rag_config} is not a valid YAML file") from e
                # try:
                #     with open(args.graph_rag_prompt_config, mode="r", encoding="utf8") as c:
                #         graph_rag_prompt_config = yaml.safe_load(c)
                # except Exception as e:
                #     self.stop(True)
                #     raise ValueError(
                #         f"{args.graph_rag_prompt_config} is not a valid YAML file"
                #     ) from e
                if bool(os.getenv("NVIDIA_API_KEY")) is True:
                    config["api_key"] = os.getenv("NVIDIA_API_KEY")
                else:
                    config["api_key"] = "NOAPIKEYSET"
                config["milvus_db_host"] = args.milvus_db_host
                config["milvus_db_port"] = args.milvus_db_port
                self._ca_rag_config = config
                self._ctx_mgr = ContextManager(
                    config=config, req_info=None, log_level=self._args.log_level
                )
            except Exception as e:
                self.stop(True)
                logger.error(traceback.format_exc())
                raise (ValueError("CA-RAG setup failed.")) from e
        else:
            self._ctx_mgr = None

        self._rails_config = None
        if not self._args.disable_guardrails:
            with TimeMeasure("GuardRails model load"):
                from nemoguardrails import LLMRails, RailsConfig

                # Load guardrails config from file
                self._rails_config = RailsConfig.from_path(args.guardrails_config)

                # Try running a test guardrails query. Error out if the query fails.
                response = LLMRails(self._rails_config).generate(
                    messages=[{"role": "user", "content": "Hi"}]
                )
                if "an internal error has occurred" in response["content"]:
                    self.stop(True)
                    raise Exception("Guardrails failed")
                logger.info("Loaded Guardrails")

        if "ENABLE_VIA_HEALTH_EVAL" in os.environ and bool(os.environ["ENABLE_VIA_HEALTH_EVAL"]):
            self._via_health_eval = True

        logger.info("Initialized VIA Stream Handler")

    def _start_ca_rag_alert_handler(self):
        app = FastAPI()

        @app.post("/via-alert-callback")
        async def handle_alert(data: dict):
            print(json.dumps(data, indent=2))
            title = data["title"]
            message = data["message"]
            doc_meta = data["metadata"]
            with self._lock:
                alert = self._alert_info_map.get(doc_meta["event_id"], None)
            if alert:
                await alert.alert_tool.notify(title, message, doc_meta)

        config = uvicorn.Config(app, host="127.0.0.1", port=ALERT_CALLBACK_PORT)
        self._ca_rag_alert_handler_server = uvicorn.Server(config)

        self._ca_rag_alert_handler_thread = Thread(
            target=self._ca_rag_alert_handler_server.run, daemon=True
        )
        self._ca_rag_alert_handler_thread.start()

    def _process_output(
        self,
        req_info: RequestInfo,
        is_live_stream_ended: bool,
        chunk_responses: list[VlmChunkResponse],
    ):
        new_response = []
        if not is_live_stream_ended and req_info.status != RequestInfo.Status.FAILED:
            try:
                new_response = self._get_aggregated_summary(req_info, chunk_responses)
            except Exception as ex:
                logger.error("".join(traceback.format_exception(ex)))
                if not req_info.is_live:
                    req_info.status = RequestInfo.Status.FAILED
                else:
                    req_info.response += [
                        RequestInfo.Response(
                            chunk_responses[0].chunk.start_ntp,
                            chunk_responses[-1].chunk.end_ntp,
                            "Summarization failed",
                        )
                    ]
            req_info.response += new_response

        if req_info.is_live:
            live_stream_id = req_info.assets[0].asset_id
            if new_response:
                logger.info(
                    "Generated new summary for live stream %s request %s,"
                    " start-time %s end-time %s",
                    live_stream_id,
                    req_info.request_id,
                    new_response[0].start_timestamp,
                    new_response[-1].end_timestamp,
                )
            elif chunk_responses:
                logger.error(
                    "Failed to generate summary for live stream %s request %s,"
                    " start-time %s end-time %s",
                    live_stream_id,
                    req_info.request_id,
                    chunk_responses[0].chunk.start_ntp,
                    chunk_responses[-1].chunk.end_ntp,
                )

            if is_live_stream_ended:
                if live_stream_id in self._live_stream_info_map:
                    lsinfo = self._live_stream_info_map[live_stream_id]
                    lsinfo.live_stream_ended = True
                    if not lsinfo.stop:
                        concurrent.futures.wait(lsinfo.pending_futures)
                req_info.end_time = time.time()
                req_info.progress = 100
                req_info.status = RequestInfo.Status.SUCCESSFUL
                self._metrics.active_live_streams.dec()
                self.stop_via_gpu_monitor(req_info, chunk_responses)
        else:
            if req_info.status == RequestInfo.Status.FAILED:
                logger.info(
                    "Summary generation failed for video file request %s", req_info.request_id
                )
                self.stop_via_gpu_monitor(req_info, chunk_responses)
            else:
                req_info.progress = 100
                req_info.end_time = time.time()
                self.stop_via_gpu_monitor(req_info, chunk_responses)
                req_info.status = RequestInfo.Status.SUCCESSFUL
                cuda.cudart.cudaProfilerStop()
                nvtx.end_range(req_info.nvtx_summarization_start)
                logger.info(
                    "Summary generated for video file request %s,"
                    " total processing time - %.2f seconds, summary %s",
                    req_info.request_id,
                    req_info.end_time - req_info.start_time,
                    "",
                )

            # Unlock the asset and update metrics
            for asset in req_info.assets:
                asset.unlock()
            self._metrics.queries_processed.inc()
            self._metrics.queries_pending.dec()

    def _get_cv_metadata_for_chunk(self, json_file, frame_times):
        cv_meta = []
        if json_file:
            with open(json_file, "r") as f:
                data = json.load(f)
            for frame_time in frame_times:
                # Get the corresponding entry in cv metadata
                # BN : TBD : optimize this to do incremental search
                # instead of searching complete meta every time
                for item in data:
                    item_time = item["timestamp"] / 1e9
                    if item_time >= 0.99 * frame_time and item_time <= 1.01 * frame_time:
                        cv_meta.append(item)
                        break
        return cv_meta

    def _on_vlm_chunk_response(self, response: VlmChunkResponse, req_info: RequestInfo):
        """Gather chunks processed by the pipeline and run any further post-processing"""
        chunk = response.chunk
        vlm_response = response.vlm_response
        frame_times = response.frame_times

        if response.error:
            if not req_info.is_live:
                # Error was encountered while processing a chunk,
                # mark the request as failed for files
                # For live streams, continue processing new chunks
                req_info.status = RequestInfo.Status.FAILED
                self._vlm_pipeline.abort_chunks(req_info.assets[0].asset_id)
            logger.error(
                "Encountered error while processing chunk %r of query %s - %s",
                chunk,
                req_info.request_id,
                response.error,
            )
        elif vlm_response is not None:
            response.vlm_response = vlm_response
            # Add the chunk VLM response to the milvus DB
            if req_info._ctx_mgr:
                # Along with chunk, add cv metadata for the chunk
                # get cv metadata present in file req_info.cv_metadata_json_file
                # for duration chunk.start_pts to chunk.end_pts
                cv_meta = self._get_cv_metadata_for_chunk(
                    req_info.cv_metadata_json_file, frame_times
                )
                print(
                    f"chunkIdx = {chunk.chunkIdx}  chunk.start_pts = {chunk.start_pts} \
                      chunk.end_pts = {chunk.end_pts} CV metadata length = {len(cv_meta)}"
                )
                with TimeMeasure("Context Manager - Add Doc"):
                    add_doc_start_time = time.time()
                    req_info._ctx_mgr.add_doc(
                        vlm_response,
                        doc_i=chunk.chunkIdx,
                        doc_meta=(
                            vars(chunk)
                            | {"request_id": req_info.request_id, "cv_meta": json.dumps(cv_meta)}
                        ),
                        callback=lambda output: logger.debug(
                            f"Summary till now: {output.result()}"
                        ),
                    )
                    if req_info.last_chunk is None or req_info.last_chunk.chunkIdx < chunk.chunkIdx:
                        req_info.last_chunk = chunk
                    add_doc_end_time = time.time()
                    response.add_doc_start_time = add_doc_start_time
                    response.add_doc_end_time = add_doc_end_time
        if req_info.is_live:
            live_stream_id = req_info.assets[0].asset_id
            lsinfo = self._live_stream_info_map[live_stream_id]

            if not response.is_live_stream_ended:
                logger.info(
                    "Generated new response for live-stream %s, query %s, chunk %r, summary %s",
                    live_stream_id,
                    req_info.request_id,
                    chunk,
                    vlm_response,
                )
                req_info.processed_chunk_list.append(response)
                req_info.chunk_count += 1

            req_info.processed_chunk_list.sort(key=lambda x: x.chunk.chunkIdx)

            gathered_chunks = 0
            gathered_chunks_total_duration = 0

            summ_batch_size = req_info.summary_duration // req_info.chunk_size
            if req_info.processed_chunk_list:
                curIdx = req_info.processed_chunk_list[0].chunk.chunkIdx
                gathered_chunks = 1

                for processed_chunk in req_info.processed_chunk_list[1:]:
                    if processed_chunk.chunk.chunkIdx != curIdx + 1:
                        break
                    curIdx += 1
                    gathered_chunks += 1
                    if gathered_chunks == summ_batch_size:
                        break

            # Calculate the total duration of gathered chunks
            gathered_chunks_total_duration = (
                ntp_to_unix_timestamp(
                    req_info.processed_chunk_list[gathered_chunks - 1].chunk.end_ntp
                )
                - ntp_to_unix_timestamp(req_info.processed_chunk_list[0].chunk.start_ntp)
                if req_info.processed_chunk_list
                else 0
            )

            logger.info(
                "Gathered %d chunks, total chunk duration \
                    is %.2f sec for query %s, summary duration %d sec",
                gathered_chunks,
                gathered_chunks_total_duration,
                req_info.request_id,
                req_info.summary_duration,
            )

            if (
                (
                    req_info.summary_duration == 0
                    or req_info._ctx_mgr is None
                    or gathered_chunks == req_info.summary_duration // req_info.chunk_size
                    or response.is_live_stream_ended
                )
                and gathered_chunks > 0
                and not lsinfo.stop
            ):
                if response.is_live_stream_ended and req_info.last_chunk is not None:
                    last_chunk = req_info.last_chunk.model_copy(deep=True)
                    last_chunk.start_ntp = last_chunk.end_ntp
                    last_chunk.start_ntp_float = last_chunk.end_ntp_float
                    last_chunk.start_pts = last_chunk.end_pts
                    last_chunk.chunkIdx = last_chunk.chunkIdx + 1
                    last_chunk.is_last = True
                    last_meta = vars(last_chunk)
                    last_meta["cv_meta"] = ""
                    last_meta["request_id"] = req_info.request_id
                    req_info._ctx_mgr.add_doc(
                        ".",
                        doc_i=last_chunk.chunkIdx,
                        doc_meta=last_meta,
                    )
                # Summary Duration not specified or total duration is greater than summary duration.
                logger.info(
                    "Generating summary for live stream %s request %s",
                    live_stream_id,
                    req_info.request_id,
                )

                if len(lsinfo.pending_futures) > 1:
                    logger.warning(
                        "Possible high load on the system detected. This may result in higher"
                        " response times. Try reducing number of streams or increasing the chunk"
                        " size or tuning the CA-RAG config for reduced latency."
                    )

                # Theo: req_info.processed_chunk_list is a list of vlm_response, outputs of VLM.
                fut = req_info._output_process_thread_pool.submit(
                    self._process_output,
                    req_info,
                    False,
                    req_info.processed_chunk_list[:gathered_chunks],
                )  
                lsinfo.pending_futures.append(fut)

                def handle_future_done(fut: concurrent.futures.Future):
                    if fut.exception():
                        logger.error("".join(traceback.format_exception(fut.exception())))

                fut.add_done_callback(handle_future_done)
                fut.add_done_callback(lsinfo.pending_futures.remove)
                req_info.processed_chunk_list = req_info.processed_chunk_list[gathered_chunks:]

            if response.is_live_stream_ended:
                if lsinfo.stop:
                    req_info.status = RequestInfo.Status.STOPPING
                    for fut in lsinfo.pending_futures:
                        fut.cancel()

                # Queue that the request be marked completed
                # once all pending aggregation requests are completed.
                fut = req_info._output_process_thread_pool.submit(
                    self._process_output, req_info, True, []
                )
                fut.add_done_callback(
                    lambda fut, tpool=req_info._output_process_thread_pool: tpool.shutdown(
                        wait=False
                    )
                )
            return

        # Cache the processed chunk of a file
        req_info.processed_chunk_list.append(response)
        req_info.progress = 90 * len(req_info.processed_chunk_list) / req_info.chunk_count
        logger.info(
            "Processed chunk for query %s, total chunks %d, processed chunks %d, chunk %r,",
            req_info.request_id,
            req_info.chunk_count,
            len(req_info.processed_chunk_list),
            chunk,
        )

        if len(req_info.processed_chunk_list) == req_info.chunk_count:
            # All chunks of file processed
            nvtx.end_range(req_info.nvtx_vlm_start)
            cur_time = time.time()

            if req_info.status == RequestInfo.Status.FAILED:
                self._vlm_pipeline.abort_chunks_done(req_info.assets[0].asset_id)
            else:
                logger.info(
                    "Processed all chunks for query %s, VLM pipeline time %.2f sec",
                    req_info.request_id,
                    cur_time - req_info.start_time,
                )
                logger.info("Generating summary for request %s", req_info.request_id)
                if req_info._health_summary:
                    req_info._health_summary.vlm_pipeline_latency = cur_time - req_info.start_time

            # Queue for getting the aggregated summary
            req_info._output_process_thread_pool.submit(
                self._process_output, req_info, False, req_info.processed_chunk_list
            )
            req_info._output_process_thread_pool.shutdown(wait=False)

    def _trigger_query(self, req_info: RequestInfo):
        """Trigger a query on a file"""
        from file_splitter import FileSplitter

        logger.info("Triggering oldest queued query %s", req_info.request_id)
        req_info.status = RequestInfo.Status.PROCESSING
        req_info.start_time = time.time()

        # Trigger collecting VIA GPU health metrics
        self.start_via_gpu_monitor(req_info)

        if req_info._ctx_mgr:
            ca_rag_config = copy.deepcopy(self._ca_rag_config)

            if req_info.caption_summarization_prompt:
                ca_rag_config["summarization"]["prompts"][
                    "caption"
                ] = req_info.vlm_request_params.vlm_prompt
                ca_rag_config["summarization"]["prompts"][
                    "caption_summarization"
                ] = req_info.caption_summarization_prompt
            if req_info.summary_aggregation_prompt:
                ca_rag_config["summarization"]["prompts"][
                    "summary_aggregation"
                ] = req_info.summary_aggregation_prompt

            if req_info.is_live:
                ca_rag_config["summarization"]["params"]["batch_size"] = int(
                    req_info.summary_duration / req_info.chunk_size
                )
            if req_info.summarize_batch_size:
                ca_rag_config["summarization"]["params"][
                    "batch_size"
                ] = req_info.summarize_batch_size
            if req_info.rag_type:
                ca_rag_config["chat"]["rag"] = req_info.rag_type
            if req_info.rag_batch_size:
                ca_rag_config["chat"]["params"]["batch_size"] = req_info.rag_batch_size
            if req_info.rag_top_k:
                ca_rag_config["chat"]["params"]["top_k"] = req_info.rag_top_k

            # TODO: if the prompt is changing then what do we reset?
            # TODO: This needs to be fixed
            # How should we reset this?
            req_info._ctx_mgr.reset(expr="pk > 0")
            if req_info.summarize:
                logger.debug(f"Updating Context Manager with config {ca_rag_config}")
                req_info._ctx_mgr.update(ca_rag_config)
        else:
            logger.debug("Request does not contain Context Manager")

        def _on_new_chunk(chunk: ChunkInfo):
            """Callback for when a new chunk is created"""
            if chunk is None:
                return
            chunk.streamId = req_info.assets[0].asset_id
            self._vlm_pipeline.enqueue_chunk(
                chunk,
                lambda response, req_info=req_info: self._on_vlm_chunk_response(response, req_info),
                req_info.vlm_request_params,
                req_info.num_frames_per_chunk,
                req_info.vlm_input_width,
                req_info.vlm_input_height,
            )
            req_info.chunk_count += 1

        # Set start/end times if not specified by user
        if not req_info.start_timestamp:
            req_info.start_timestamp = 0
        if req_info.end_timestamp is None:
            req_info.end_timestamp = req_info.file_duration / 1e9

        enable_dense_caption = bool(os.environ.get("ENABLE_DENSE_CAPTION", False))
        if enable_dense_caption:
            # Get dense caption from file if present
            saved_dc_file = req_info.file + ".dc.json"
            if os.access(saved_dc_file, os.R_OK):
                logger.info(f"Saved DC available {saved_dc_file}")
                req_info_deserialized = DCSerializer.from_json(saved_dc_file)
                req_info.chunk_count = len(req_info_deserialized.processed_chunk_list)
                for vlm_response in req_info_deserialized.processed_chunk_list:
                    self._on_vlm_chunk_response(vlm_response, req_info)
                return

        # Create virtual file chunks
        paths_string = ";".join([asset.path for asset in req_info.assets])
        nvtx_file_split_start = nvtx.start_range(
            message="File Splitting-" + str(req_info.request_id), color="blue"
        )
        FileSplitter(
            paths_string,
            FileSplitter.SplitMode.SEEK,
            req_info.chunk_size,
            start_pts=int(req_info.start_timestamp * 1e9),
            end_pts=int(req_info.end_timestamp * 1e9),
            sliding_window_overlap_sec=req_info.chunk_overlap_duration,
            on_new_chunk=lambda chunk: _on_new_chunk(chunk),
        ).split()
        nvtx.end_range(nvtx_file_split_start)

        # No chunks were created. Mark the request completed and trigger next query if queued
        if req_info.chunk_count == 0:
            req_info.status = RequestInfo.Status.SUCCESSFUL
            req_info.progress = 100
            req_info.end_time = time.time()
            req_info.response = []
        req_info.nvtx_vlm_start = nvtx.start_range(
            message="VLM Pipeline-" + str(req_info.request_id), color="green"
        )

    def get_ctx_mgr(self, assets: list[Asset]) -> ContextManager | None:
        """
        Return a ContextManager associated with the given assets.
        """
        with self._lock:
            for _, request_info in self._request_info_map.items():
                req_matches = True
                for asset in assets:
                    if asset not in request_info.assets:
                        req_matches = False
                        break
                if req_matches:
                    return request_info._ctx_mgr

        return None

    def get_request_infos(self, assets: list[Asset]) -> list[RequestInfo]:
        """
        Returns a list of request_infos associated with the given assets.

        Args:
            assets (list[Asset]): A list of Asset objects to find the request_infos for.

        Returns:
            list[RequestInfo]: A list of request_infos associated with the assets
        """
        with self._lock:
            request_infos = []
            for asset in assets:
                for request_id, request_info in self._request_info_map.items():
                    if asset in request_info.assets:
                        request_infos.append(request_info)
        return request_infos

    def qa(
        self,
        assets: list[Asset],
        messages: str = None,
        generation_config=None,
        start_timestamp=None,
        end_timestamp=None,
    ):
        try:
            request_infos = self.get_request_infos(assets)
            if len(request_infos) > 1:
                logger.info(
                    f"Multiple video processing requests identified for same assets;"
                    f" using request to identify the Graph database: {str(request_infos[-1])}"
                )
            if len(request_infos) >= 1:
                if request_infos[-1].enable_chat is False:
                    return (
                        "Chat functionality disabled for request id: "
                        + request_infos[-1].request_id
                    )

                # Run guardrails on the user supplied prompt
                if self._rails_config:
                    with TimeMeasure("Guardrails process"):
                        logger.info("Guardrails in progress")
                        from nemoguardrails import LLMRails

                        response = LLMRails(self._rails_config).generate(
                            messages=[{"role": "user", "content": messages}]
                        )
                        if response["content"] != "lmm":
                            logger.info("Guardrails engaged")
                            return response["content"]
                        logger.info("Guardrails pass")
                chat_config = self._ca_rag_config["chat"]
                if chat_config["rag"] != "vector-rag" and chat_config["rag"] != "graph-rag":
                    logger.info("Both graph rag and vector rag are disabled. Q&A is disabled.")
                    return "Both graph rag and vector rag are disabled. Q&A is disabled."
                else:
                    result = request_infos[-1]._ctx_mgr.call(
                        {
                            "chat": {
                                "question": messages,
                                "is_live": request_infos[-1].is_live,
                                "is_last": False,
                            }
                        }
                    )
                    logger.debug(f"Q&A: result object is {result}")
                    return result["chat"]["response"]
            else:
                return (
                    "Chat functionality disabled; "
                    "please call /summarize API with enable_chat: True;"
                )
        except Exception as e:
            error_message = f"An error occurred: {str(e)} - {e.__class__.__name__}"
            logger.error(error_message)
            raise ViaException(error_message)

    def summarize(
        self,
        assets: list[Asset],
        prompt: str = None,
        chunk_size=0,
        chunk_overlap_duration=0,
        generation_config=None,
        start_timestamp=None,
        end_timestamp=None,
        caption_summarization_prompt="",
        summary_aggregation_prompt="",
        summarize=None,
        enable_chat=True,
        enable_cv_metadata=True,
        graph_rag_prompt_yaml="",
        num_frames_per_chunk=0,
        summarize_batch_size=None,
        rag_type=None,
        rag_top_k=None,
        rag_batch_size=None,
        vlm_input_width=0,
        vlm_input_height=0,
    ):
        """Run a summarization query on a file"""
        # Enable summarization if summarization config is enabled  OR API passes enable flag
        # Enable summarization if none provided
        if self._ctx_mgr:
            summarize_enable = self._ca_rag_config.get("summarization", {})
            summarize_enable = summarize_enable.get("enable", True)
            if summarize is None:
                summarize = summarize_enable
        cuda.cudart.cudaProfilerStart()
        if prompt:
            summarization_query = prompt
        else:
            summarization_query = self.default_caption_prompt
        return self.query(
            assets=assets,
            query=summarization_query,
            chunk_size=chunk_size,
            chunk_overlap_duration=chunk_overlap_duration,
            generation_config=generation_config,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            is_summarization=True,
            caption_summarization_prompt=caption_summarization_prompt,
            summary_aggregation_prompt=summary_aggregation_prompt,
            summarize=summarize,
            enable_chat=enable_chat,
            enable_cv_metadata=enable_cv_metadata,
            graph_rag_prompt_yaml=graph_rag_prompt_yaml,
            num_frames_per_chunk=num_frames_per_chunk,
            summarize_batch_size=summarize_batch_size,
            rag_top_k=rag_top_k,
            rag_type=rag_type,
            rag_batch_size=rag_batch_size,
            vlm_input_width=vlm_input_width,
            vlm_input_height=vlm_input_height,
        )

    def query(
        self,
        assets: list[Asset],
        query: str,
        chunk_size=0,
        chunk_overlap_duration=0,
        generation_config=None,
        start_timestamp=None,
        end_timestamp=None,
        is_summarization=False,
        caption_summarization_prompt="",
        summary_aggregation_prompt="",
        summarize=None,
        enable_chat=True,
        enable_cv_metadata=True,
        graph_rag_prompt_yaml="",
        num_frames_per_chunk=0,
        summarize_batch_size=None,
        rag_type=None,
        rag_top_k=None,
        rag_batch_size=None,
        vlm_input_width=0,
        vlm_input_height=0,
    ):
        """Run a query on a file"""

        if self._args.disable_ca_rag is True and (enable_chat is True):
            raise ViaException("CA-RAG must be enabled to use chat feature", "BadParameter", 400)

        try:
            # Get file duration
            file_duration = MediaFileInfo.get_info(assets[0].path).video_duration_nsec
        except gi.repository.GLib.GError as ex:
            raise ViaException(ex.message, "FailedRequest", 400)

        if (
            self._args.max_file_duration != 0
            and file_duration > self._args.max_file_duration * 60000000000
        ):
            return (
                False,
                f"File duration {round(file_duration/60000000000, 2)} is greater"
                f" than max allowed {self._args.max_file_duration} minutes",
                None,
            )

        if chunk_size > 0 and chunk_overlap_duration > 0 and chunk_overlap_duration >= chunk_size:
            raise ViaException(
                "chunkOverlapDuration must be less than chunkDuration", "BadParameter", 400
            )

        # Run guardrails on the user supplied prompt
        if self._rails_config:
            with TimeMeasure("Guardrails process"):
                nvtx_guardrails_start = nvtx.start_range(message="Guardrails-", color="blue")
                from nemoguardrails import LLMRails

                # Load guardrails config from file
                response = LLMRails(self._rails_config).generate(
                    messages=[{"role": "user", "content": query}]
                )
                nvtx.end_range(nvtx_guardrails_start)
                if response["content"] != "lmm":
                    if "an internal error has occurred" in response["content"]:
                        logger.error("Guardrails failed")
                        raise ViaException("An internal error has occurred")
                    raise ViaException(response["content"], "", 400)

        # Create a RequestInfo object and populate it
        req_info = RequestInfo()
        req_info.file = assets[0].path
        req_info.chunk_size = chunk_size
        req_info.is_summarization = is_summarization
        req_info.vlm_request_params.vlm_prompt = query
        req_info.vlm_request_params.vlm_generation_config = generation_config
        req_info.assets = assets
        req_info.start_timestamp = start_timestamp
        req_info.end_timestamp = end_timestamp
        req_info.file_duration = file_duration
        req_info.summary_aggregation_prompt = summary_aggregation_prompt
        req_info.caption_summarization_prompt = caption_summarization_prompt
        req_info.graph_rag_prompt_yaml = graph_rag_prompt_yaml
        req_info._output_process_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        req_info.summarize = summarize
        req_info.enable_chat = enable_chat
        req_info.num_frames_per_chunk = num_frames_per_chunk
        req_info.summarize_batch_size = summarize_batch_size
        req_info.rag_type = rag_type
        req_info.rag_top_k = rag_top_k
        req_info.rag_batch_size = rag_batch_size
        req_info.vlm_input_width = vlm_input_width
        req_info.vlm_input_height = vlm_input_height

        # FIXME(shaunakg/slakhotia): How do we handle this in the new design?
        req_info.nvtx_summarization_start = nvtx.start_range(
            message="Summarization-" + str(req_info.request_id), color="blue"
        )
        if not self._args.disable_ca_rag:
            req_info._ctx_mgr = self.get_ctx_mgr(req_info.assets)
            if not req_info._ctx_mgr:
                req_info._ctx_mgr = ContextManager(
                    config=self._ca_rag_config, req_info=req_info, log_level=self._args.log_level
                )

        # Lock the asset(s) so that it cannot be deleted while it is being used.
        for asset in req_info.assets:
            asset.lock()

        req_info.chunk_overlap_duration = chunk_overlap_duration

        req_info.queue_time = time.time()
        with self._lock:
            self._request_info_map[req_info.request_id] = req_info

        # Add the request to the pending queue
        self._metrics.queries_pending.inc()

        self._trigger_query(req_info)

        return req_info.request_id

    def start_via_gpu_monitor(self, req_info):
        # Start collecting VIA GPU health metrics if enabled
        if self._via_health_eval and req_info._monitor is None:
            logger.info(f"Starting GPUMonitor for request {req_info.request_id}")
            req_info._monitor = GPUMonitor()
            req_info._monitor.start_recording_nvdec(
                interval_in_seconds=0.2,
                nvdec_plot_file_name="/tmp/via-logs/via_nvdec_usage_"
                + str(req_info.request_id)
                + ".csv",
            )
            req_info._monitor.start_recording_gpu_usage(
                interval_in_seconds=0.2,
                gpu_plot_file_name="/tmp/via-logs/via_gpu_usage_"
                + str(req_info.request_id)
                + ".csv",
            )
            req_info._health_summary = RequestHealthMetrics()
            req_info._health_summary.health_graph_paths = [
                req_info._monitor.nvdec_plot_file_name,
                req_info._monitor.gpu_plot_file_name,
            ]
            req_info._health_summary.set_gpu_names(req_info._monitor.get_gpu_names())
            req_info._health_summary.chunk_size = req_info.chunk_size
            req_info._health_summary.chunk_overlap_duration = req_info.chunk_overlap_duration
            req_info._health_summary.input_video_duration = req_info.file_duration / (
                1000 * 1000 * 1000
            )  # ns to s
            if req_info._health_summary.chunk_size <= 0:
                req_info._health_summary.num_chunks = 1
            else:
                req_info._health_summary.num_chunks = (
                    req_info._health_summary.input_video_duration
                    / req_info._health_summary.chunk_size
                )
            req_info._health_summary.num_gpus = self._args.num_gpus
            info = self.get_models_info()
            req_info._health_summary.vlm_model_name = str(info.id)
            req_info._health_summary.vlm_batch_size = self._args.vlm_batch_size

    def stop_via_gpu_monitor(self, req_info, chunk_responses: list[VlmChunkResponse]):
        if self._via_health_eval and req_info._monitor is not None:
            logger.info(f"Stopping GPUMonitor for request {req_info.request_id}")
            plot_graph_file = "/tmp/via-logs/via_plot_nvdec_" + str(req_info.request_id) + ".png"
            plot_graph_files = {
                "gpu": "/tmp/via-logs/via_plot_gpu_" + str(req_info.request_id) + ".png",
                "gpu_mem": "/tmp/via-logs/via_plot_gpu_mem_" + str(req_info.request_id) + ".png",
            }
            req_info._monitor.stop_recording_nvdec(plot_graph_file=plot_graph_file)
            req_info._monitor.stop_recording_gpu(plot_graph_files=plot_graph_files)
            req_info._health_summary.health_graph_plot_paths = [
                plot_graph_file,
                plot_graph_files["gpu"],
                plot_graph_files["gpu_mem"],
            ]
            req_info._health_summary.e2e_latency = time.time() - req_info.start_time

            def find_extreme(responses, func, value):
                values = []
                for response in responses:
                    if hasattr(response, value):
                        attr_value = getattr(response, value)
                        if attr_value is not None:
                            values.append(attr_value)
                if not values:
                    return 0
                return func(values)

            if chunk_responses:
                max_decode_end_time = find_extreme(chunk_responses, max, "decode_end_time")
                min_decode_start_time = find_extreme(chunk_responses, min, "decode_start_time")
                req_info._health_summary.decode_latency = (
                    max_decode_end_time - min_decode_start_time
                )
                max_vlm_end_time = find_extreme(chunk_responses, max, "vlm_end_time")
                min_vlm_embed_start_time = find_extreme(chunk_responses, min, "embed_start_time")
                if min_vlm_embed_start_time == 0:
                    # embed_start_time unavailable, use vlm_start_time instead
                    min_vlm_embed_start_time = find_extreme(chunk_responses, min, "vlm_start_time")
                req_info._health_summary.vlm_latency = max_vlm_end_time - min_vlm_embed_start_time
                req_info._health_summary.pending_doc_start_time = (
                    req_info.pending_add_doc_start_time
                )
                req_info._health_summary.pending_doc_end_time = req_info.pending_add_doc_end_time
                req_info._health_summary.pending_add_doc_latency = (
                    req_info.pending_add_doc_end_time - req_info.pending_add_doc_start_time
                )
                req_info._health_summary.req_start_time = req_info.start_time
                req_info._health_summary.total_vlm_input_tokens = sum(
                    [resp.vlm_stats.get("input_tokens", 0) for resp in chunk_responses]
                )
                req_info._health_summary.total_vlm_output_tokens = sum(
                    [resp.vlm_stats.get("output_tokens", 0) for resp in chunk_responses]
                )
                try:
                    for response in chunk_responses:
                        req_info._health_summary.all_times.append(
                            {
                                "chunk_id": response.chunk.chunkIdx,
                                "decode_start": response.decode_start_time,
                                "decode_end": response.decode_end_time,
                                "embed_start": response.embed_start_time,
                                "embed_end": response.embed_end_time,
                                "vlm_start": response.vlm_start_time,
                                "vlm_end": response.vlm_end_time,
                                "add_doc_start": response.add_doc_start_time,
                                "add_doc_end": response.add_doc_end_time,
                                "vlm_stats": response.vlm_stats,
                            }
                        )
                except Exception as e:
                    print("Error:", e)
            req_info._health_summary.ca_rag_latency = req_info._ca_rag_latency
            logger.debug(f"_health_summary json: {str(vars(req_info._health_summary))}")
            health_summary_file_name = (
                "/tmp/via-logs/via_health_summary_" + str(req_info.request_id) + ".json"
            )
            req_info._health_summary.dump_json(file_name=health_summary_file_name)
            logger.info(f"VIA Health Summary written to {health_summary_file_name}")
            req_info._monitor = None

    def add_rtsp_stream(self, asset: Asset, chunk_size=None):
        """Add an RTSP stream to the server and start streaming

        Args:
            asset: Live stream asset to add
            chunk_size: Chunk size to use, in seconds
        """

        # A live stream can be added only once
        with self._lock:
            if asset.asset_id in self._live_stream_info_map:
                raise ViaException(
                    "Live stream already has query "
                    f"'{self._live_stream_info_map[asset.asset_id].req_info[0].request_id}' running."  # noqa: E501
                    " Update or stop the same query.",
                    "BadParameters",
                    400,
                )

            if len(self._live_stream_info_map) >= self._args.max_live_streams:
                raise ViaException(
                    "Server is already processing maximum number of live streams"
                    f" ({self._args.max_live_streams})",
                    503,
                )

            if chunk_size is None or chunk_size == 0:
                raise ViaException(
                    "Non-zero chunk duration required for live-stream", "InvalidParameter", 400
                )

            # Create a live stream info object and populate it
            live_stream_info = LiveStreamInfo()
            live_stream_info.chunk_size = chunk_size
            live_stream_info.asset = asset

            # Lock the asset so that it cannot be deleted while it is being used.
            asset.lock()

            self._live_stream_info_map[asset.asset_id] = live_stream_info

    def add_rtsp_stream_query(
        self,
        asset: Asset,
        query: str,
        chunk_duration: int = 0,
        generation_config=None,
        summary_duration=0,
        caption_summarization_prompt="",
        summary_aggregation_prompt="",
        summarize=True,
        enable_chat=True,
        enable_cv_pipeline=True,
        graph_rag_prompt_yaml="",
        num_frames_per_chunk=0,
        vlm_input_width=0,
        vlm_input_height=0,
        summarize_batch_size=None,
        rag_type=None,
        rag_top_k=None,
        rag_batch_size=None,
    ):
        """Add a query on the RTSP stream

        Args:
            asset: Asset to add the query on
            query: VLM query prompt
            generation_config: VLM generation configuration.
            summary_duration: Summarization duration, in seconds.
                              Defaults to 0 (summarize each chunk separately).
            caption_summarization_prompt: LLM prompt to use to extract summary from VLM response.
            summary_aggregation_prompt: LLM prompt to use to aggregate summaries of
                                        individual chunks.

        Returns:
            A unique ID for the request
        """
        live_stream_info = self._live_stream_info_map[asset.asset_id]
        if len(live_stream_info.req_info) > 0:
            raise ViaException(
                "Live stream already has query "
                f"'{live_stream_info.req_info[0].request_id}' running."
                " Update or stop the same query.",
                "BadParameters",
                400,
            )

        if summary_duration == 0:
            raise ViaException("summary_duration must be non-zero", "BadParameters", 400)

        if summary_duration % chunk_duration != 0:
            raise ViaException(
                "summary_duration must be an exact multiple of chunk_duration", "BadParameters", 400
            )

        # Highest preference is to the user specified VLM prompt in the API call,
        # next to the VLM prompt (caption) in the CA RAG config. Lastly to the
        # prompt specified as argument to the app
        if not query:
            query = self.default_caption_prompt

        # Run guardrails on the user supplied prompt
        if self._rails_config:
            with TimeMeasure("Guardrails process"):
                from nemoguardrails import LLMRails

                response = LLMRails(self._rails_config).generate(
                    messages=[{"role": "user", "content": query}]
                )
                if response["content"] != "lmm":
                    raise ViaException(response["content"], "", 400)

        # Create a RequestInfo object and populate it
        req_info = RequestInfo()
        req_info.file = asset.path
        req_info.chunk_size = chunk_duration
        req_info.is_summarization = True
        req_info.vlm_request_params.vlm_prompt = query
        req_info.vlm_request_params.vlm_generation_config = generation_config
        req_info.is_live = True
        req_info.status = RequestInfo.Status.PROCESSING
        req_info.summary_duration = summary_duration
        req_info.start_time = time.time()
        req_info.queue_time = time.time()
        req_info.assets = [asset]
        req_info.summary_aggregation_prompt = summary_aggregation_prompt
        req_info.caption_summarization_prompt = caption_summarization_prompt
        req_info.graph_rag_prompt_yaml = graph_rag_prompt_yaml
        req_info._output_process_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if self._ctx_mgr:
            summarize_enable = self._ca_rag_config.get("summarization", {})
            summarize_enable = summarize_enable.get("enable", True)
            if summarize is None:
                summarize = summarize_enable
        req_info.summarize = summarize
        req_info.enable_chat = enable_chat
        req_info.num_frames_per_chunk = num_frames_per_chunk
        req_info.vlm_input_width = vlm_input_width
        req_info.vlm_input_height = vlm_input_height
        req_info.summarize_batch_size = summarize_batch_size
        req_info.rag_type = rag_type
        req_info.rag_top_k = rag_top_k
        req_info.rag_batch_size = rag_batch_size

        if not self._args.disable_ca_rag:
            req_info._ctx_mgr = self.get_ctx_mgr(req_info.assets)
            if not req_info._ctx_mgr:
                req_info._ctx_mgr = ContextManager(
                    config=self._ca_rag_config, req_info=req_info, log_level=self._args.log_level
                )
            ca_rag_config = copy.deepcopy(self._ca_rag_config)

            if req_info.caption_summarization_prompt:
                ca_rag_config["summarization"]["prompts"][
                    "caption"
                ] = req_info.vlm_request_params.vlm_prompt
                ca_rag_config["summarization"]["prompts"][
                    "caption_summarization"
                ] = req_info.caption_summarization_prompt
            if req_info.summary_aggregation_prompt:
                ca_rag_config["summarization"]["prompts"][
                    "summary_aggregation"
                ] = req_info.summary_aggregation_prompt

            if req_info.is_live:
                ca_rag_config["summarization"]["params"]["batch_size"] = int(
                    req_info.summary_duration / req_info.chunk_size
                )
            if req_info.summarize_batch_size:
                ca_rag_config["summarization"]["params"][
                    "batch_size"
                ] = req_info.summarize_batch_size
            if req_info.rag_type:
                ca_rag_config["chat"]["rag"] = req_info.rag_type
            if req_info.rag_batch_size:
                ca_rag_config["chat"]["params"]["batch_size"] = req_info.rag_batch_size
            if req_info.rag_top_k:
                ca_rag_config["chat"]["params"]["top_k"] = req_info.rag_top_k
            req_info._ctx_mgr.update(ca_rag_config)
            req_info._ctx_mgr.reset(expr="pk > 0")

        # Add the request to the request info map
        with self._lock:
            self._request_info_map[req_info.request_id] = req_info

        live_stream_info.req_info.append(req_info)
        self._metrics.active_live_streams.inc()

        # Trigger collecting VIA GPU health metrics
        self.start_via_gpu_monitor(req_info)

        self._vlm_pipeline.add_live_stream(
            asset.asset_id,
            asset.path,
            live_stream_info.chunk_size,
            lambda response, req_info=req_info: self._on_vlm_chunk_response(response, req_info),
            req_info.vlm_request_params,
            username=asset.username,
            password=asset.password,
            num_frames_per_chunk=num_frames_per_chunk,
            vlm_input_width=vlm_input_width,
            vlm_input_height=vlm_input_height,
        )

        return req_info.request_id

    def remove_rtsp_stream_query(self, request_id):
        """Remove a VLM query from the RTSP stream"""
        with self._lock:
            asset_id = self._request_info_map[request_id].asset.asset_id
            self._live_stream_info_map[asset_id].req_info.pop()

    # Reset CA RAG: milvus, GraphRAG DB
    # To be used for NVCF deployement on each new add /files request
    # to make sure we dont have stale data from a previous run
    # TODO: Need design discussion
    def reset_ca_rag(self):
        with self._lock:
            for req_info in self._request_info_map.values():
                logger.info(f"Resetting CA RAG for {req_info.request_id}")
                if req_info._ctx_mgr:
                    req_info._ctx_mgr.reset(expr="pk > 0")
                    del req_info._ctx_mgr
                    req_info._ctx_mgr = None
                    req_info.assets = []

    def remove_video_file(self, asset: Asset):
        logger.info("Removing video %s from pipeline", asset.asset_id)
        ctx_mgrs_to_be_removed = []
        with self._lock:
            for req_info in self._request_info_map.values():
                if asset in req_info.assets and req_info._ctx_mgr:
                    ctx_mgrs_to_be_removed.append(req_info._ctx_mgr)
                    req_info._ctx_mgr = None

            self._request_info_map = {
                req_id: req_info
                for req_id, req_info in self._request_info_map.items()
                if asset not in req_info.assets
            }

        for ctx_mgr in ctx_mgrs_to_be_removed:
            ctx_mgr.reset(expr="pk > 0")
            del ctx_mgr

    def remove_rtsp_stream(self, asset: Asset):
        """Remove an RTSP stream from the server"""
        with self._lock:
            if asset.asset_id not in self._live_stream_info_map:
                logger.debug(f"RTSP stream for video {asset.asset_id} not active")
                return
            logger.info("Removing live stream %s from pipeline", asset.asset_id)
            live_stream_info = self._live_stream_info_map[asset.asset_id]
        live_stream_info.stop = True

        self._vlm_pipeline.remove_live_stream(asset.asset_id)

        # Unlock the asset so that it may be deleted and remove the stream
        # from live stream info map
        live_stream_info.asset.unlock()

        with self._lock:
            for alert_id in list(self._alert_info_map.keys()):
                if self._alert_info_map[alert_id].liveStreamId == asset.asset_id:
                    self.remove_live_stream_alert(alert_id)

            self._live_stream_info_map.pop(asset.asset_id)

        logger.info("Removed live stream %s from pipeline", asset.asset_id)

        ctx_mgrs_to_be_removed = []
        with self._lock:
            for req_info in self._request_info_map.values():
                if asset in req_info.assets and req_info._ctx_mgr:
                    ctx_mgrs_to_be_removed.append(req_info._ctx_mgr)
                    req_info._ctx_mgr = None
            self._request_info_map = {
                req_id: req_info
                for req_id, req_info in self._request_info_map.items()
                if asset not in req_info.assets
            }
        for ctx_mgr in ctx_mgrs_to_be_removed:
            ctx_mgr.reset(expr="pk > 0")
            del ctx_mgr

    def get_event_list(self, liveStreamId: str):
        events_list = []
        with self._lock:
            for alert_id, ainfo in self._alert_info_map.items():
                if ainfo.liveStreamId == liveStreamId:
                    events_list.append({"event_id": alert_id, "event_list": ainfo.events})
        return events_list

    def add_live_stream_alert(
        self,
        liveStreamId: str,
        events: list[str],
        isCallback=False,
        callbackUrl: str = "",
        callbackJsonTemplate: str = "",
        callbackToken=None,
        alertName="",
    ):
        if not self._ctx_mgr:
            raise ViaException("Alerts functionality is disabled", "MethodNotAllowed", 405)

        with self._lock:
            if liveStreamId not in self._live_stream_info_map:
                raise ViaException(
                    f"No such live-stream {liveStreamId} or live-stream not active",
                    "BadParameters",
                    400,
                )
            req_info = self._live_stream_info_map[liveStreamId].req_info[0]

        ainfo = AlertInfo()
        ainfo.liveStreamId = liveStreamId
        ainfo.events = events
        ainfo.callbackUrl = callbackUrl
        if callbackJsonTemplate:
            ainfo.callbackJsonTemplate = callbackJsonTemplate
        ainfo.callbackToken = callbackToken

        try:
            test_json = jinja2.Template(ainfo.callbackJsonTemplate).render(
                streamId=ainfo.liveStreamId,
                alertId=ainfo.alert_id,
                ntpTimestamp="1970-01-01T00:00:00.000Z",
                alertText="Some text",
                detectedEvents=json.dumps(["some event1", "some event2"]),
            )

            json.loads(test_json)
        except json.decoder.JSONDecodeError:
            raise ViaException(
                f"Json template results into invalid json '{test_json}'",
                "BadParameters",
                400,
            )

        ainfo.alert_tool = (
            AlertCallbackTool(name="alert-" + ainfo.alert_id, alert_info=ainfo)
            if isCallback
            else AlertSseTool(
                name="alert-" + ainfo.alert_id, req_info=req_info, sse_tool_name=alertName
            )
        )
        with self._lock:
            self._alert_info_map[ainfo.alert_id] = ainfo

        req_info._ctx_mgr.update({"notification": {"events": self.get_event_list(liveStreamId)}})

        return ainfo

    def remove_live_stream_alert(self, alert_id: str):
        with self._lock:
            if alert_id not in self._alert_info_map:
                raise ViaException(f"No such alert {alert_id}", "BadParameters", 400)
            ainfo = self._alert_info_map.pop(alert_id)

            liveStreamId = ainfo.liveStreamId
            if liveStreamId not in self._live_stream_info_map:
                return

            lsinfo = self._live_stream_info_map[liveStreamId]

        if lsinfo.req_info:
            if lsinfo.req_info[0]._ctx_mgr:
                lsinfo.req_info[0]._ctx_mgr.update(
                    {"notification": {"events": self.get_event_list(liveStreamId)}}
                )
        logger.info("Removed alert %s for live stream %s", alert_id, lsinfo.asset.asset_id)

    def live_stream_alerts(self):
        with self._lock:
            return list(self._alert_info_map.values())

    def add_alert(
        self,
        requestId: str,
        assetId: str,
        events: list[str],
        isCallback=False,
        callbackUrl: str = "",
        callbackJsonTemplate: str = "",
        callbackToken=None,
        alertName="",
    ):
        if not self._ctx_mgr:
            raise ViaException("Alerts functionality is disabled", "MethodNotAllowed", 405)

        with self._lock:
            if requestId not in self._request_info_map:
                raise ViaException(
                    f"No such request {requestId} or request not active",
                    "BadParameters",
                    400,
                )
            req_info = self._request_info_map[requestId]

        ainfo = AlertInfo()
        ainfo.requestId = requestId
        ainfo.liveStreamId = assetId
        ainfo.events = events
        ainfo.callbackUrl = callbackUrl
        if callbackJsonTemplate:
            ainfo.callbackJsonTemplate = callbackJsonTemplate
        ainfo.callbackToken = callbackToken

        try:
            test_json = jinja2.Template(ainfo.callbackJsonTemplate).render(
                streamId=ainfo.liveStreamId,
                alertId=ainfo.alert_id,
                ntpTimestamp="1970-01-01T00:00:00.000Z",
                alertText="Some text",
                detectedEvents=json.dumps(["some event1", "some event2"]),
            )

            json.loads(test_json)
        except json.decoder.JSONDecodeError:
            raise ViaException(
                f"Json template results into invalid json '{test_json}'",
                "BadParameters",
                400,
            )

        ainfo.alert_tool = (
            AlertCallbackTool(name="alert-" + ainfo.alert_id, alert_info=ainfo)
            if isCallback
            else AlertSseTool(
                name="alert-" + ainfo.alert_id, req_info=req_info, sse_tool_name=alertName
            )
        )

        with self._lock:
            self._alert_info_map[ainfo.alert_id] = ainfo

        req_info._ctx_mgr.update({"notification": {"events": self.get_event_list(assetId)}})

        return ainfo

    def remove_alert(self, alert_id: str):
        with self._lock:
            if alert_id not in self._alert_info_map:
                raise ViaException(f"No such alert {alert_id}", "BadParameters", 400)
            ainfo = self._alert_info_map.pop(alert_id)

            requestId = ainfo.requestId
            if requestId not in self._request_info_map:
                return

            req_info = self._request_info_map[requestId]

        if req_info._ctx_mgr:
            req_info._ctx_mgr.update(
                {"notification": {"events": self.get_event_list(ainfo.liveStreamId)}}
            )
        logger.info("Removed alert %s for live stream %s", alert_id, req_info.assets[0].asset_id)

    def stop(self, force=False):
        """Stop the VIA Stream Handler"""
        logger.info("Stopping VIA Stream Handler")
        self._vlm_pipeline.stop(force)

        self._metrics.unregister()

        self._summary_db_handler = None
        self._ctx_mgr = None

        logger.info("Stopped VIA Stream Handler")

    def get_response(self, request_id, chunk_response_size=None):
        """Get currently available response for the request

        Args:
            request_id: ID of the request
            chunk_response_size: Number of chunked responses to include.
                                 Defaults to None (all available).

        Returns:
            A tuple of the request details and currently available response
        """
        with self._lock:
            if request_id not in self._request_info_map:
                raise ViaException(f"No such request-id {request_id}", "InvalidParameterValue", 400)

            req_info = self._request_info_map[request_id]
        if chunk_response_size is None:
            # Return all available response
            response = req_info.response
            # Reset response to empty
            req_info.response = []
        else:
            # Get user specified number of chunked responses
            response = req_info.response[:chunk_response_size]
            # Remove the responses that will be returned
            req_info.response = req_info.response[chunk_response_size:]
        return req_info, response

    async def wait_for_request_done(self, request_id):
        """Wait for request to either complete or fail."""

        with self._lock:
            if request_id not in self._request_info_map:
                raise ViaException(f"No such request-id {request_id}", "InvalidParameterValue", 400)
            req_info = self._request_info_map[request_id]

        while req_info.status not in [RequestInfo.Status.FAILED, RequestInfo.Status.SUCCESSFUL]:
            logger.info(
                "Status for query %s is %s, percent complete is %.2f, size of response list is %d",
                req_info.request_id,
                req_info.status.value,
                req_info.progress,
                len(req_info.response),
            )
            await asyncio.sleep(1)

    def get_models_info(self):
        return self._vlm_pipeline.get_models_info()

    def _get_aggregated_summary(
        self, req_info: RequestInfo, chunk_responses: list[VlmChunkResponse]
    ):
        """Aggregated summary for the request"""

        with nvtx.annotate(message="StreamHandler/SaveDCFile", color="yellow"):
            saved_dc_file = req_info.file + ".dc.json"
            if not os.access(saved_dc_file, os.R_OK) and self._args.enable_dev_dc_gen:
                logger.info(f"Generating DC file at {saved_dc_file}")
                # Serialize the object to a JSON file
                req_info_to_write = req_info
                DCSerializer.to_json(req_info_to_write, saved_dc_file)

        if chunk_responses:
            with nvtx.annotate(message="StreamHandler/FilterNSort", color="yellow"):
                # Filter out chunks that do not have an associated vlm response
                chunk_responses = list(
                    filter(lambda item: item.vlm_response is not None, chunk_responses)
                )
                # Sort chunks based on their start times
                chunk_responses.sort(key=lambda item: ntp_to_unix_timestamp(item.chunk.start_ntp))

        if len(chunk_responses) == 0:
            # Return empty response if there are no chunks / chunks with vlm responses
            return []

        if self._via_health_eval is True:
            with open("/tmp/via-logs/vlm_testdata_" + str(req_info.request_id) + ".txt", "w") as f:
                with nvtx.annotate(message="StreamHandler/WriteChnkIDAns", color="green"):
                    f.write("Chunk_ID,Answer\n")
                    for proc_chunk in chunk_responses:
                        idx = proc_chunk.chunk.chunkIdx
                        summ = proc_chunk.vlm_response.replace("\n", "  ")
                        f.write(f'{idx},"{summ}"\n')

        if req_info._ctx_mgr:
            with TimeMeasure("Context Manager Summarize") as cms_t:
                try:
                    with nvtx.annotate(
                        message="CA RAG-" + str(req_info.request_id), color="yellow"
                    ):
                        # Summarize indivudual chunk VLM responses using CA-RAG
                        # TODO: Handle the last chunk id, should be -1
                        if not req_info.is_live:
                            last_meta = vars(chunk_responses[-1].chunk)
                            last_meta["is_last"] = True
                            last_meta["request_id"] = req_info.request_id
                            last_meta["cv_meta"] = ""
                            with TimeMeasure("Context Manager Summarize/call"):
                                req_info._ctx_mgr.add_doc(
                                    ".",
                                    doc_i=chunk_responses[-1].chunk.chunkIdx + 1,
                                    doc_meta=last_meta,
                                )
                        if req_info.summarize:
                            if req_info.enable_chat:
                                with TimeMeasure("Context Manager Summarize/summarize"):
                                    agg_response = req_info._ctx_mgr.call(
                                        {
                                            "summarization": {
                                                "start_index": chunk_responses[0].chunk.chunkIdx,
                                                "end_index": chunk_responses[-1].chunk.chunkIdx,
                                            },
                                            "chat": {"post_process": True},
                                        }
                                    )
                            else:
                                with TimeMeasure("Context Manager Summarize/summarize"):
                                    agg_response = req_info._ctx_mgr.call(
                                        {
                                            "summarization": {
                                                "start_index": chunk_responses[0].chunk.chunkIdx,
                                                "end_index": chunk_responses[-1].chunk.chunkIdx,
                                            }
                                        }
                                    )
                            agg_response = agg_response["summarization"]["result"]
                            if self._via_health_eval is True:
                                with open(
                                    "/tmp/via-logs/summ_testdata_"
                                    + str(req_info.request_id)
                                    + ".txt",
                                    "w",
                                ) as f:
                                    f.write("Chunk_ID,Answer\n")
                                    summ = str(agg_response).replace("\n", "  ")
                                    f.write(f'{0},"{summ}"\n')
                        else:
                            agg_response = "Media processed"
                except Exception as ex:
                    logger.error(traceback.format_exc())
                    logger.error(
                        "Summary aggregation failed for query %s - %s", req_info.request_id, str(ex)
                    )
                    agg_response = "Something went wrong. Try again!\n"
            req_info._ca_rag_latency = cms_t.execution_time

            # Return summarized response
            return [
                RequestInfo.Response(
                    (
                        chunk_responses[0].chunk.start_ntp
                        if req_info.is_live
                        else chunk_responses[0].chunk.start_pts / 1e9
                    ),
                    (
                        chunk_responses[-1].chunk.end_ntp
                        if req_info.is_live
                        else chunk_responses[-1].chunk.end_pts / 1e9
                    ),
                    agg_response,
                )
            ]

        # CA-RAG is disabled. Return a list of individual chunk VLM responses
        return [
            RequestInfo.Response(
                (
                    processed_chunk.chunk.start_ntp
                    if req_info.is_live
                    else processed_chunk.chunk.start_pts / 1e9
                ),
                (
                    processed_chunk.chunk.end_ntp
                    if req_info.is_live
                    else processed_chunk.chunk.end_pts / 1e9
                ),
                processed_chunk.vlm_response,
            )
            for processed_chunk in chunk_responses
        ]

    @staticmethod
    def populate_argument_parser(parser: ArgumentParser):
        """Add VIA Stream Handler arguments to the argument parser"""

        VlmPipeline.populate_argument_parser(parser)

        parser.add_argument(
            "--disable-guardrails",
            action="store_true",
            default=False,
            help="Disable NEMO Guardrails",
        )
        parser.add_argument(
            "--enable-dev-dc-gen",
            action="store_true",
            default=False,
            help="Disable NEMO Guardrails",
        )
        parser.add_argument(
            "--disable-cv-pipeline",
            action="store_true",
            default=False,
            help="Disable CV Pipeline",
        )
        parser.add_argument(
            "--guardrails-config",
            type=str,
            default="/opt/nvidia/via/guardrails_config",
            help="NEMO Guardrails configuration",
        )
        parser.add_argument(
            "--max-file-duration",
            type=int,
            default=0,
            help="Maximum file duration to allow (0 = no restriction)",
        )

        parser.add_argument(
            "--milvus-db-port",
            type=str,
            default="19530",
            help="Port to use Milvus DB on",
        )
        parser.add_argument(
            "--milvus-db-host",
            type=str,
            default="127.0.0.1",
            help="Host to use Milvus DB on",
        )
        parser.add_argument(
            "--disable-ca-rag",
            action="store_true",
            default=False,
            help="Enable/Disable CA-RAG",
        )
        parser.add_argument(
            "--ca-rag-config",
            type=str,
            default="/opt/nvidia/via/default_config.yaml",
            help="CA RAG config path",
        )
        parser.add_argument(
            "--graph-rag-prompt-config",
            type=str,
            default="/opt/nvidia/via/warehouse_graph_rag_config.yaml",
            help="Graph RAG prompt config path",
        )
        parser.add_argument(
            "--summarization-query",
            type=str,
            default="Summarize the video",
            help="LLM query to use for summarization",
        )
        parser.add_argument(
            "--asset-dir", type=str, help="Directory to store the assets in", default="assets"
        )


def handle_rtsp_input(stream_handler: ViaStreamHandler, args):
    asset_id = str(uuid.uuid4())
    asset_dir = "/tmp/via/" + asset_id
    os.makedirs(asset_dir)
    asset = Asset(asset_id, args.file_or_rtsp, args.file_or_rtsp, "", "", asset_dir)
    stream_handler.add_rtsp_stream(asset)

    try:
        req_id = stream_handler.add_rtsp_stream_query(asset, args.summarization_query)
    except ViaException as ex:
        logger.error(f"Video query failed: {ex.message}")
        os.system(f"rm -rf {asset_dir}")
        return
    except Exception:
        logger.error("Video query failed")
        os.system(f"rm -rf {asset_dir}")
        return

    try:
        while True:
            req_info, response = stream_handler.get_response(req_id)
            print_response(req_info, response)
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    stream_handler.remove_rtsp_stream_query(req_id)
    stream_handler.remove_rtsp_stream(asset)
    os.system(f"rm -rf {asset_dir}")


def handle_file(stream_handler: ViaStreamHandler, args):
    asset_id = str(uuid.uuid4())
    asset_dir = "/tmp/via/" + asset_id
    os.makedirs(asset_dir)
    cur_time = datetime.now(timezone.utc)
    cur_timestamp = (
        cur_time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(cur_time.timestamp() * 1000) % 1000:03d}Z"
    )
    asset = Asset(
        asset_id,
        args.file_or_rtsp,
        os.path.basename(args.file_or_rtsp),
        "",
        cur_timestamp,
        asset_dir,
    )

    try:
        req_id = stream_handler.summarize(asset, chunk_size=args.chunk_size)
    except ViaException as ex:
        logger.error(f"Video query failed: {ex.message}")
        os.system(f"rm -rf {asset_dir}")
        return
    except Exception:
        logger.error("Video query failed")
        os.system(f"rm -rf {asset_dir}")
        return

    req_info, response = stream_handler.get_response(req_id)
    while req_info.status in [RequestInfo.Status.PROCESSING, RequestInfo.Status.QUEUED]:
        time.sleep(1)
        req_info, response = stream_handler.get_response(req_id)
    if not print_response(req_info, response):
        os.system(f"rm -rf {asset_dir}")
        return

    if args.interactive_qa:
        while True:
            try:
                print("Enter your query: ", end="")
                query = input()
                if not query:
                    continue
            except KeyboardInterrupt:
                print("Stopping Q&A")
                break

            try:
                req_id = stream_handler.query(asset, query, chunk_size=args.chunk_size)
            except ViaException as ex:
                logger.error(f"Video query failed: {ex.message}")
                continue
            except Exception:
                logger.error("Video query failed")
                continue

            req_info, response = stream_handler.get_response(req_id)
            while req_info.status in [RequestInfo.Status.PROCESSING, RequestInfo.Status.QUEUED]:
                time.sleep(1)
                req_info, response = stream_handler.get_response(req_id)
            print_response(req_info, response)

    os.system(f"rm -rf {asset_dir}")


def print_response(req_info: RequestInfo, response: list[RequestInfo.Response]):
    if (
        req_info.status in [RequestInfo.Status.PROCESSING, RequestInfo.Status.SUCCESSFUL]
        and response
    ):
        print(
            tabulate(
                [
                    [
                        textwrap.fill(
                            f"{item.start_timestamp} -> {item.end_timestamp}",
                            width=24,
                            replace_whitespace=False,
                            drop_whitespace=False,
                        ),
                        textwrap.fill(
                            item.response,
                            width=(shutil.get_terminal_size().columns - 31),
                            replace_whitespace=False,
                            drop_whitespace=False,
                        ),
                    ]
                    for item in response
                ],
                tablefmt="simple_grid",
            )
        )
        return True
    if req_info.status == RequestInfo.Status.FAILED:
        logger.error("Video query failed")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VIA Stream Handler", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ViaStreamHandler.populate_argument_parser(parser)
    parser.add_argument("--chunk-size", default=0, type=int, help="Chunk size in seconds")
    parser.add_argument(
        "--interactive-qa",
        action="store_true",
        default=False,
        help="Start interactive Q&A after initial embedding generation",
    )
    parser.add_argument(
        "--aggregate-responses",
        action="store_true",
        default=False,
        help="Wether to aggregate individual chunk summaries",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["error", "warn", "info", "debug"],
        default="info",
        help="Application log level",
    )
    parser.add_argument("file_or_rtsp", type=str, help="File or RTSP stream to run the VIA on")
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())
    logging.basicConfig(level=args.log_level.upper())

    try:
        with TimeMeasure("Total Load"):
            stream_handler = ViaStreamHandler(args)
    except Exception as ex:
        logger.error("Could not load VIA Stream Handler - " + str(ex))
        sys.exit(-1)

    if args.file_or_rtsp.startswith("rtsp://"):
        handle_rtsp_input(stream_handler, args)
    else:
        handle_file(stream_handler, args)

    stream_handler.stop()
