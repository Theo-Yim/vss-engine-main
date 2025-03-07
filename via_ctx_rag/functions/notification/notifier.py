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
import json

from via_ctx_rag.base import Function
from via_ctx_rag.tools.notification import NotificationTool
from via_ctx_rag.utils.ctx_rag_logger import logger, TimeMeasure
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence


class Notifier(Function):
    """Refine Summarization Function"""

    prompt_str: str = 'You are a Event Detection System. From the following\
                        text, detect if any of the following events are present {events}. \
                        The output should be a json in the format:\
                        {{\
                            "result" : [\
                                {{\
                                    "event": "$event_name_1",\
                                    "is_detected": $is_detected\
                                }},\
                                {{\
                                    "event": "$event_name_2",\
                                    "is_detected": $is_detected\
                                }}\
                                {{\
                                    "event": "$event_name_n",\
                                    "is_detected": $is_detected\
                                }}\
                            ]\
                        }}\
                        where "$event_name_i" is the ith event to be detected and\
                        "$is_detected" is either true or false.\
                        Output should only be the said json; nothing else!'
    output_parser = StrOutputParser()
    pipeline: RunnableSequence
    notification_tool: NotificationTool
    events: list[dict]

    def setup(self):
        self.notification_tool = self.get_tool("notification_tool")
        self.events = (
            self.get_param("events", required=False)
            if self.get_param("events", required=False)
            else []
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.prompt_str), ("user", "{input}")]
        )
        self.output_parser = StrOutputParser()
        self.pipeline = self.prompt | self.get_tool("llm") | self.output_parser

    async def acall(self, state: dict):
        return await asyncio.sleep(0.001)

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        # TODO: Ensure the output is proper. Ensure json load works.
        notifications = []
        tasks = []

        with TimeMeasure("notifier/llm_call"):
            for event_item in self.events:
                event = event_item["event_list"]
                event_id = event_item["event_id"]

                tasks.append(
                    self.pipeline.ainvoke(
                        {
                            "events": str(event),
                            "input": doc,
                        }
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

        with TimeMeasure("notifier/notify_call"):
            for result, event_item in zip(results, self.events):
                event_id = event_item["event_id"]

                if isinstance(result, Exception):
                    logger.warning(
                        f"Pipeline invocation failed for event_id {event_id}: {result}"
                    )
                    continue

                result = result.strip("`json")
                try:
                    result = json.loads(result)
                except Exception:
                    logger.warning(
                        f"Notification failed due to incorrect json generation:\n{result}"
                    )
                    continue

                events_detected = [
                    item["event"] for item in result["result"] if item["is_detected"]
                ]

                if events_detected:
                    events_detected_str = " ".join(events_detected)
                    notifications.append(
                        self.notification_tool.notify(
                            title=f"{events_detected_str} detected!",
                            message=f"{events_detected_str} detected in '{doc}'",
                            metadata=doc_meta
                            | {
                                "doc": doc,
                                "events_detected": events_detected,
                                "event_id": event_id,
                            },
                        )
                    )

            if notifications:
                await asyncio.gather(*notifications)
