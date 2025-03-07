################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def get_timestamp_str(ts):
    """Get RFC3339 string timestamp"""
    return (
        datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        + f".{(int(ts * 1000) % 1000):03d}Z"
    )


class ChunkInfo(BaseModel):
    """Represents a video chunk"""

    streamId: str = Field(default="")
    chunkIdx: int = Field(default=0)
    file: str = Field(default="")
    pts_offset_ns: int = Field(default=0)
    start_pts: int = Field(default=0)
    end_pts: int = Field(default=-1)
    start_ntp: str = Field(default="")
    end_ntp: str = Field(default="")
    start_ntp_float: float = Field(default=0.0)
    end_ntp_float: float = Field(default=0.0)
    is_first: bool = Field(default=False)
    is_last: bool = Field(default=False)

    def __repr__(self) -> str:
        if self.file.startswith("rtsp://"):
            return (
                f"Chunk {self.chunkIdx}: start={self.start_pts / 1000000000.0}"
                f" end={self.end_pts / 1000000000.0} start_ntp={self.start_ntp}"
                f" end_ntp={self.end_ntp} file={self.file}"
            )
        return (
            f"Chunk {self.chunkIdx}: start={self.start_pts / 1000000000.0}"
            f" end={self.end_pts / 1000000000.0} file={self.file}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def get_timestamp(self, frame_pts) -> str:
        timestamp_str = ""
        if self.file.startswith("rtsp://"):
            timestamp_float = self.start_ntp_float + frame_pts - self.start_pts / 1000000000.0
            timestamp_str = get_timestamp_str(timestamp_float)
        else:
            timestamp_str = str(frame_pts)
        return timestamp_str
