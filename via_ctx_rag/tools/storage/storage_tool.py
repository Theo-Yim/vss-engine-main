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

"""storage_handler.py:"""

from via_ctx_rag.base import Tool


class StorageTool(Tool):
    def __init__(self, name="storage_tool") -> None:
        super().__init__(name)

    def add_summary(self, summary, metadata):
        pass

    async def aadd_summary(self, summary, metadata):
        pass

    def add_summaries(self, batch_summary, batch_metadata):
        pass

    async def aadd_summaries(self, batch_summary, batch_metadata):
        pass

    def get_text_data(self, fields, filter):
        pass

    async def aget_text_data(self, fields, filter):
        pass

    def search(self, search_query):
        pass
