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

"""notification_tool.py:"""

from via_ctx_rag.base import Tool


class NotificationTool(Tool):
    def __init__(self, name="notification_tool") -> None:
        super().__init__(name)

    async def notify(self, title: str, message: str, metadata: dict):
        pass
