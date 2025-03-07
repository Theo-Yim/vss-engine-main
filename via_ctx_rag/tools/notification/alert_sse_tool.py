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

import aiohttp
from via_ctx_rag.tools.notification import NotificationTool
from via_ctx_rag.utils.ctx_rag_logger import logger


class AlertSSETool(NotificationTool):
    """Tool for sending an alert as a post request to the endpoint.
    Implements NotificationTool class
    """

    def __init__(self, endpoint: str, name="alert_sse_notifier") -> None:
        super().__init__(name)
        self.alert_endpoint = endpoint

    async def notify(self, title: str, message: str, metadata: dict):
        try:
            headers = {}
            body = {
                "title": title,
                "message": message,
                "metadata": metadata,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.alert_endpoint, json=body, headers=headers
                ) as r:
                    r.raise_for_status()
        except Exception as ex:
            events_detected = metadata.get("events_detected", [])
            logger.error(
                "Alert callback failed for event(s) '%s' - %s",
                ", ".join(events_detected),
                str(ex),
            )
