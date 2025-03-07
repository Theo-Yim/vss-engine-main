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

from via_ctx_rag.tools.notification import NotificationTool
from via_ctx_rag.utils.ctx_rag_logger import logger


class EchoNotificationTool(NotificationTool):
    """Tool for printing notification on the terminal.
    Implements NotificationTool class
    """

    def __init__(self, name="echo_notifier") -> None:
        super().__init__(name)

    async def notify(self, title: str, message: str, metadata: dict):
        try:
            logger.info("==================Notification==================")
            logger.info(f"Notification: {title}")
            logger.info(f"Message: {message}")
            logger.info(f"Metadata: {metadata}")
            logger.info("================================================")
            return True
        except Exception:
            logger.error("Echo Notification Failed")
            return False
