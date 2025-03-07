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
"""Video File Frame Getter

This module supports getting frames from a video file either as raw frame tensors or
JPEG encoded images. Supports decoding of a part of file using start/end timestamps,
picking N frames from the segment as well as pre-processing the decoded frames
as required by the VLM model.
"""

import ctypes
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Callable

import cupy as cp
import gi
import gst_video_sei_meta
import numpy as np
import pyds
import torch
from torchvision.transforms import v2

from chunk_info import ChunkInfo
from utils import MediaFileInfo
from via_logger import TimeMeasure, logger

gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst  # noqa: E402

Gst.init(None)


def get_timestamp_str(ts):
    """Get RFC3339 string timestamp"""
    return (
        datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        + f".{(int(ts * 1000) % 1000):03d}Z"
    )


class ToCHW:
    """
    Converts tensor from HWC (interleaved) to CHW (planar)
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        return clip.permute(2, 0, 1)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Rescale:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0
    """

    def __init__(self, factor):
        self._factor = factor
        pass

    def __call__(self, clip):
        return clip.float().mul(self._factor)

    def __repr__(self) -> str:
        return self.__class__.__name__


class BaseFrameSelector:
    """Base Frame Selector

    Base class for implementing a frame selector."""

    def __init__(self):
        self._chunk = None

    def set_chunk(self, chunk: ChunkInfo):
        """Set Chunk to select frames from"""
        self._chunk = chunk

    def choose_frame(self, buffer, pts: int):
        """Choose a frame for processing.

        Implementations should return a boolean indicating if the frame should
        be chosen for processing.

        Args:
            buffer: GstBuffer
            pts: Frame timestamp in nanoseconds.

        Returns:
            bool: Boolean indicating if the frame should be chosen for processing.
        """
        return False


class DefaultFrameSelector:
    """Default Frame Selector.

    Selects N equally spaced frames from a chunk.
    """

    def __init__(self, num_frames=8):
        """Default initializer.

        Args:
            num_frames (int, optional): Number of frames to select from a chunk. Defaults to 8.
        """
        self._num_frames = num_frames
        self._selected_pts_array = []

    def set_chunk(self, chunk: ChunkInfo):
        self._chunk = chunk
        self._selected_pts_array = []
        start_pts = chunk.start_pts
        end_pts = chunk.end_pts

        if start_pts == -1 or end_pts == -1:
            # If start or end PTS is not set (=-1), set it to 0 and file duration
            # to decode the entire file
            start_pts = 0
            end_pts = MediaFileInfo.get_info(chunk.file).video_duration_nsec

        # Adjust for the PTS offset (in case of split files)
        start_pts -= chunk.pts_offset_ns
        end_pts -= chunk.pts_offset_ns

        # Calculate PTS for N equally spaced frames
        pts_diff = (end_pts - start_pts) / self._num_frames
        for i in range(self._num_frames):
            self._selected_pts_array.append(start_pts + i * pts_diff)
        logger.debug(f"Selected PTS = {self._selected_pts_array} for {chunk}")

    def choose_frame(self, buffer, pts):
        # Choose the frame if it's PTS is more than the next sampled PTS in the
        # list.
        if (
            len(self._selected_pts_array)
            and pts >= self._selected_pts_array[0]
            and pts <= self._chunk.end_pts
        ):
            while len(self._selected_pts_array) and pts >= self._selected_pts_array[0]:
                self._selected_pts_array.pop(0)
            return True
        if pts >= self._chunk.end_pts:
            self._selected_pts_array.clear()
        return False


class VideoFileFrameGetter:
    """Get frames from a video file as a list of tensors."""

    def __init__(
        self,
        frame_selector: BaseFrameSelector,
        frame_width=0,
        frame_height=0,
        gpu_id=0,
        do_preprocess=False,
        image_mean=[],
        rescale_factor=0,
        image_std=0,
        crop_height=0,
        crop_width=0,
        shortest_edge: int | None = None,
        enable_jpeg_output=False,
        image_aspect_ratio="",
        data_type_int8=False,
    ) -> None:
        self._selected_pts_array = []
        self._last_gst_buffer = None
        self._loop = None
        self._frame_selector = frame_selector
        self._chunk = None
        self._gpu_id = gpu_id
        self._sei_base_time = None
        self._frame_width = self._frame_width_orig = frame_width
        self._frame_height = self._frame_height_orig = frame_height
        self._uridecodebin = None
        self._image_mean = image_mean
        self._rescale_factor = rescale_factor
        self._image_std = image_std
        self._crop_height = crop_height
        self._crop_width = crop_width
        self._shortest_edge = shortest_edge
        self._do_preprocess = do_preprocess
        self._image_aspect_ratio = image_aspect_ratio
        self._enable_jpeg_output = enable_jpeg_output
        self._data_type_int8 = data_type_int8
        self._pipeline = None
        self._last_stream_id = ""
        self._is_live = False
        self._live_stream_frame_selectors: dict[BaseFrameSelector, any] = {}
        self._live_stream_frame_selectors_lock = Lock()
        self._live_stream_next_chunk_start_pts = 0
        self._live_stream_next_chunk_idx = 0
        self._live_stream_chunk_duration = 0
        self._live_stream_chunk_overlap_duration = 0
        self._live_stream_ntp_epoch = 0
        self._live_stream_ntp_pts = 0
        self._live_stream_chunk_decoded_callback: Callable[
            [ChunkInfo, torch.Tensor | list[np.ndarray], list[float]], None
        ] = None
        self._first_frame_width = 0
        self._first_frame_height = 0
        self._got_error = False
        self._previous_frame_width = 0
        self._previous_frame_height = 0
        self._destroy_pipeline = False
        self._last_frame_pts = 0
        self._uridecodebin = None
        self._rtspsrc = None
        self._udpsrc = None

    def _set_frame_resolution(self, frame_width, frame_height):
        if frame_width and (self._previous_frame_width != frame_width):
            self._previous_frame_width = self._frame_width
            self._frame_width = frame_width
            self._destroy_pipeline = True
        if frame_height and (self._previous_frame_height != frame_height):
            self._previous_frame_height = self._frame_height
            self._frame_height = frame_height
            self._destroy_pipeline = True

    def _preprocess(self, frames):
        if frames and not self._enable_jpeg_output:
            frames = torch.stack(frames)
            if not self._data_type_int8:
                frames = frames.half()
            if self._do_preprocess:
                if self._crop_height and self._crop_width:
                    frames = v2.functional.center_crop(
                        frames, [self._crop_height, self._crop_width]
                    )
                frames = v2.functional.normalize(
                    frames,
                    [x / (self._rescale_factor) for x in self._image_mean],
                    [x / (self._rescale_factor) for x in self._image_std],
                ).half()
        return frames

    def _process_finished_chunks(self, current_pts=None, flush=False):
        chunks_processed_fs = []

        for fs, (cached_pts, cached_frames) in self._live_stream_frame_selectors.items():
            if (
                (current_pts is not None and current_pts >= fs._chunk.end_pts)
                or len(fs._selected_pts_array) == 0
                or flush
            ):
                if len(cached_pts) == len(cached_frames) or flush:
                    cached_frames = self._preprocess(cached_frames)
                    base_time = (
                        self._live_stream_ntp_epoch - self._live_stream_ntp_pts
                    ) / 1000000000
                    if self._sei_base_time:
                        base_time = self._sei_base_time / 1000000000
                    if flush and self._last_frame_pts >= fs._chunk.start_pts:
                        fs._chunk.end_pts = self._last_frame_pts

                    fs._chunk.start_ntp = get_timestamp_str(base_time + fs._chunk.start_pts / 1e9)
                    fs._chunk.end_ntp = get_timestamp_str(base_time + fs._chunk.end_pts / 1e9)
                    fs._chunk.start_ntp_float = base_time + (fs._chunk.start_pts / 1e9)
                    fs._chunk.end_ntp_float = base_time + (fs._chunk.end_pts / 1e9)

                    self._live_stream_chunk_decoded_callback(fs._chunk, cached_frames, cached_pts)
                    chunks_processed_fs.append(fs)

        for fs in chunks_processed_fs:
            self._live_stream_frame_selectors.pop(fs)

    def _create_pipeline(self, file_or_rtsp: str, username="", password=""):
        # Construct DeepStream pipeline for decoding
        # For raw frames as tensor:
        # uridecodebin -> probe (frame selector) -> nvvideconvert -> appsink
        #     -> frame pre-processing -> add to cache
        # For jpeg images:
        # uridecodebin -> probe (frame selector) -> nvjpegenc -> appsink -> add to cache
        self._is_live = file_or_rtsp.startswith("rtsp://")
        pipeline = Gst.Pipeline()

        uridecodebin = Gst.ElementFactory.make("uridecodebin")
        if self._is_live:
            uridecodebin.set_property("uri", file_or_rtsp)
        else:
            uridecodebin.set_property("uri", f"file://{os.path.abspath(file_or_rtsp)}")
        pipeline.add(uridecodebin)
        self._uridecodebin = uridecodebin

        self._q1 = Gst.ElementFactory.make("queue")
        pipeline.add(self._q1)
        q2 = Gst.ElementFactory.make("queue")
        pipeline.add(q2)

        videoconvert = Gst.ElementFactory.make("nvvideoconvert")
        self._videoconvert = videoconvert
        videoconvert.set_property("nvbuf-memory-type", 2)

        videoconvert.set_property("gpu-id", self._gpu_id)
        pipeline.add(videoconvert)

        if self._enable_jpeg_output:
            jpegenc = Gst.ElementFactory.make("nvjpegenc")
            pipeline.add(jpegenc)
            format = "RGB"  # only RGB/I420 supported by nvjpegenc
        else:
            format = "GBR" if self._do_preprocess else "RGB"
            pass
        capsfilter = Gst.ElementFactory.make("capsfilter")
        self._out_caps_filter = capsfilter
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                (
                    f"video/x-raw(memory:NVMM), format={format},"
                    f" width={self._frame_width}, height={self._frame_height}"
                )
                if self._frame_width and self._frame_height
                else f"video/x-raw(memory:NVMM), format={format}"
            ),
        )
        pipeline.add(capsfilter)

        def buffer_probe(pad, info, data):
            # Probe callback function to pass chosen frames and drop other frames
            buffer = info.get_buffer()
            if buffer.pts == Gst.CLOCK_TIME_NONE:
                return Gst.PadProbeReturn.DROP

            self._last_frame_pts = buffer.pts

            if self._is_live:
                with self._live_stream_frame_selectors_lock:
                    buffer_address = hash(buffer)
                    video_sei_meta = gst_video_sei_meta.gst_buffer_get_video_sei_meta(
                        buffer_address
                    )

                    if video_sei_meta:
                        self._sei_data = json.loads(video_sei_meta.sei_metadata_ptr)
                        buffer.pts = self._sei_data["sim_time"] * 1e9
                        if self._sei_base_time is None:
                            self._sei_base_time = self._sei_data["timestamp"] - buffer.pts

                    if buffer.pts >= self._live_stream_next_chunk_start_pts:
                        fs = DefaultFrameSelector(self._frame_selector._num_frames)
                        chunk = ChunkInfo()
                        chunk.file = self._live_stream_url
                        chunk.chunkIdx = self._live_stream_next_chunk_idx
                        chunk.is_first = chunk.chunkIdx == 0
                        if chunk.is_first:
                            self._live_stream_next_chunk_start_pts = buffer.pts
                        chunk.start_pts = int(self._live_stream_next_chunk_start_pts)
                        chunk.end_pts = int(
                            chunk.start_pts + self._live_stream_chunk_duration * 1e9
                        )

                        fs.set_chunk(chunk)
                        self._live_stream_frame_selectors[fs] = ([], [])
                        self._live_stream_next_chunk_start_pts = (
                            chunk.end_pts - self._live_stream_chunk_overlap_duration * 1e9
                        )
                        self._live_stream_next_chunk_idx += 1

                    choose_frame = False
                    for fs, (
                        cached_pts,
                        cached_frames,
                    ) in self._live_stream_frame_selectors.items():
                        if fs.choose_frame(buffer, buffer.pts):
                            choose_frame = True
                            cached_pts.append(buffer.pts / 1e9)

                    self._process_finished_chunks(buffer.pts)

                if choose_frame:
                    return Gst.PadProbeReturn.OK

            else:
                if self._frame_selector.choose_frame(buffer, buffer.pts):
                    return Gst.PadProbeReturn.OK
                if len(self._frame_selector._selected_pts_array) == 0:
                    self._pipeline.send_event(Gst.Event.new_eos())

            return Gst.PadProbeReturn.DROP

        def add_to_cache(buffer, width, height):
            # Probe callback to add raw frame / jpeg image to cache
            _, mapinfo = buffer.map(Gst.MapFlags.READ)
            if self._enable_jpeg_output:
                # Buffer contains JPEG image, add to cache as is
                image_tensor = np.frombuffer(mapinfo.data, dtype=np.uint8)
            else:
                # Buffer contains raw frame

                # Extract GPU memory pointer and create tensor from it using
                # DeepStream Python Bindings and cupy
                _, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(buffer), 0)
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                owner = None
                c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
                unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
                memptr = cp.cuda.MemoryPointer(unownedmem, 0)
                n_frame_gpu = cp.ndarray(
                    shape=shape, dtype=np.uint8, memptr=memptr, strides=strides, order="C"
                )
                image_tensor = torch.tensor(
                    n_frame_gpu, dtype=torch.uint8, requires_grad=False, device="cuda"
                )

            # Cache the pre-processed frame / jpeg and its timestamp. Convert
            # the timestamps from nanoseconds to seconds.
            if self._is_live:
                with self._live_stream_frame_selectors_lock:
                    for _, (cached_pts, cached_frames) in self._live_stream_frame_selectors.items():
                        if buffer.pts / 1e9 in cached_pts:
                            cached_frames.append(image_tensor)
                    self._process_finished_chunks(buffer.pts)
            else:
                self._cached_frames.append(image_tensor)
                self._cached_frames_pts.append((buffer.pts) / 1000000000.0)
            buffer.unmap(mapinfo)
            logger.debug(f"Picked buffer {buffer.pts}")

        def on_new_sample(appsink):
            # Appsink callback to pull frame from the pipeline
            sample = appsink.emit("pull-sample")
            caps = sample.get_caps()
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")
            if self._first_frame_width == 0:
                logger.debug(f"first width,height in chunk={width}, {height}")
                self._first_frame_width = width
                self._first_frame_height = height
            if sample:
                buffer = sample.get_buffer()
                add_to_cache(buffer, width, height)
            return Gst.FlowReturn.OK

        def cb_ntpquery(pad, info, data):
            # Probe callback to handle NTP information from RTSP stream
            # This requires RTSP Sender Report support in the source.
            query = info.get_query()
            if query.type == Gst.QueryType.CUSTOM:
                struct = query.get_structure()
                if "nvds-ntp-sync" == struct.get_name():
                    _, data._live_stream_ntp_epoch = struct.get_uint64("ntp-time-epoch-ns")
                    _, data._live_stream_ntp_pts = struct.get_uint64("frame-timestamp")
            return Gst.PadProbeReturn.OK

        appsink = Gst.ElementFactory.make("appsink")
        appsink.set_property("async", False)
        appsink.set_property("sync", False)
        appsink.set_property("enable-last-sample", False)
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", on_new_sample)
        pipeline.add(appsink)

        def cb_newpad(uridecodebin, uridecodebin_pad, self):
            uridecodebin_pad.link(self._q1.get_static_pad("sink"))

        uridecodebin.connect("pad-added", cb_newpad, self)

        def cb_autoplug_continue(bin, pad, caps, udata):
            # Ignore audio
            return not caps.to_string().startswith("audio/")

        uridecodebin.connect("autoplug-continue", cb_autoplug_continue, None)

        def cb_select_stream(source, idx, caps):
            if "audio" in caps.to_string():
                return False
            return True

        def cb_before_send(rtspsrc, message, selff):
            """
            Callback function for the 'before-send' signal.

            This function is called before each RTSP request is sent. It checks if the
            message is a PAUSE command. If it is, the function returns False to skip
            sending the message. Otherwise, it returns True to allow the message to be sent.
            Skipping all msgs including: GstRtsp.RTSPMessage.PAUSE
            """
            logger.debug(f"selff._stop_stream = {selff._stop_stream}")
            if selff._stop_stream:
                logger.debug(
                    f"Intercepting stream:{message} as we are trying to move pipeline to NULL"
                )
                return False  # Skip sending the PAUSE message
            return True  # Allow sending the message

        def cb_elem_added(elem, username, password, selff):
            if "nvv4l2decoder" in elem.get_factory().get_name():
                elem.set_property("gpu-id", self._gpu_id)
                elem.set_property("extract-sei-type5-data", True)
                elem.set_property("sei-uuid", "NVDS_CUSTOMMETA")
                elem.set_property("low-latency-mode", True)
            if "rtspsrc" == elem.get_factory().get_name():
                selff._rtspsrc = elem
                pyds.configure_source_for_ntp_sync(hash(elem))
                timeout = int(os.environ.get("VSS_RTSP_TIMEOUT", "") or "2000") * 1000
                latency = int(os.environ.get("VSS_RTSP_LATENCY", "") or "2000")
                elem.set_property("timeout", timeout)
                elem.set_property("latency", latency)
                # Below code need additional review and tests.
                # Also is a feature - to let users change protocol.
                # Protocols: Allowed lower transport protocols
                # Default: 0x00000007, "tcp+udp-mcast+udp"
                # protocols = int(os.environ.get("VSS_RTSP_PROTOCOLS", "") or "7")
                # elem.set_property("protocols", protocols)

                if username and password:
                    elem.set_property("user-id", username)
                    elem.set_property("user-pw", password)

                # Ignore audio
                elem.connect("select-stream", cb_select_stream)

                # Connect before-send to handle TEARDOWN per:
                # Unfortunately, going to the NULL state involves going through PAUSED,
                # so rtspsrc does not know the difference and will send a PAUSE
                # when you wanted a TEARDOWN. The workaround is to
                # hook into the before-send signal and return FALSE in this case.
                # Source: https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html
                elem.connect("before-send", cb_before_send, selff)
            if "udpsrc" == elem.get_factory().get_name():
                logger.debug("udpsrc created")
                selff._udpsrc = elem

        uridecodebin.connect(
            "deep-element-added",
            lambda bin, subbin, elem, username=username, password=password, selff=self: cb_elem_added(  # noqa: E501
                elem, username, password, selff
            ),
        )

        pad = videoconvert.get_static_pad("sink")

        def buffer_probe_event(pad, info, data):
            # Probe callback function to pass chosen frames and drop other frames
            event = info.get_event()
            if event.type != Gst.EventType.CAPS:
                return Gst.PadProbeReturn.OK

            caps = event.parse_caps()
            struct = caps.get_structure(0)
            _, width = struct.get_int("width")
            _, height = struct.get_int("height")

            out_pad_width = 0
            out_pad_height = 0

            if self._image_aspect_ratio == "pad":
                pad_size = abs(width - height) // 2
                out_pad_width = pad_size if width < height else 0
                out_pad_height = pad_size if width > height else 0

            out_width = width + 2 * out_pad_width
            out_height = height + 2 * out_pad_height

            if self._shortest_edge is not None:
                shortest_edge = (
                    self._shortest_edge
                    if isinstance(self._shortest_edge, list)
                    else [self._shortest_edge, self._shortest_edge]
                )
                out_pad_width *= shortest_edge[0] / out_width
                out_pad_height *= shortest_edge[1] / out_height
                out_width, out_height = shortest_edge

            self._out_caps_filter.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw(memory:NVMM), format=GBR, width={out_width}, height={out_height}"
                ),
            )

            if out_pad_width or out_pad_height:
                self._videoconvert.set_property(
                    "dest-crop",
                    (
                        f"{int(out_pad_width)}:{int(out_pad_height)}:"
                        f"{int(out_width-2*out_pad_width)}:{int(out_height-2*out_pad_height)}"
                    ),
                )
                self._videoconvert.set_property("interpolation-method", 1)

            return Gst.PadProbeReturn.OK

        if self._do_preprocess:
            # Event probe to calculate and set pre-processing params based on file resolution
            pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, buffer_probe_event, self)

        pad.add_probe(Gst.PadProbeType.BUFFER, buffer_probe, self)
        pad.add_probe(Gst.PadProbeType.QUERY_DOWNSTREAM, cb_ntpquery, self)

        self._q1.link(videoconvert)

        videoconvert.link(capsfilter)
        if self._enable_jpeg_output:
            capsfilter.link(jpegenc)
            jpegenc.link(q2)
        else:
            capsfilter.link(q2)

        q2.link(appsink)

        self._loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        self._bus = bus

        def bus_call(bus, message, selff):
            t = message.type
            if t == Gst.MessageType.EOS:
                # sys.stdout.write("End-of-stream\n")
                selff._loop.quit()
            elif t == Gst.MessageType.WARNING:
                err, debug = message.parse_warning()

                # Ignore known harmless warnings
                if "Retrying using a tcp connection" in debug:
                    return True

                sys.stderr.write("Warning: %s: %s\n" % (err, debug))
            elif t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                sys.stderr.write("Error: %s: %s\n" % (err, debug))
                self._got_error = True
                selff._loop.quit()
            return True

        bus.connect("message", bus_call, self)
        return pipeline

    def destroy_pipeline(self):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

    def get_frames(self, chunk: ChunkInfo, retain_pipeline=False, frame_selector=None):
        """Get frames from a chunk

        Args:
            chunk (ChunkInfo): Chunk to get frames from

        Returns:
            (list[tensor], list[float]): List of tensors containing raw frames or jpeg images
                                         and a list of corresponding timestamps in seconds
        """
        self._cached_frames = []
        self._cached_frames_pts = []

        old_pipeline = None
        # ";" in chunk.file denotes a list of files
        for file in chunk.file.split(";"):
            if (
                self._last_stream_id != (chunk.streamId + file) or self._destroy_pipeline
            ) and self._pipeline:
                old_pipeline = self._pipeline
                self._pipeline = None
                self._destroy_pipeline = False
                if not (self._frame_width and self._frame_height):
                    # Next pipeline should use same resoulution as first
                    # to allow all frames in the chunk have same resoulution
                    self._frame_width = self._first_frame_width
                    self._frame_height = self._first_frame_height

            self._last_stream_id = chunk.streamId + file

            if not self._pipeline:
                self._pipeline = self._create_pipeline(file)
            pipeline = self._pipeline

            # Set start/end time in the file based on chunk info.
            frame_selector_backup = self._frame_selector
            if frame_selector:
                self._frame_selector = frame_selector
            self._frame_selector.set_chunk(chunk)
            start_pts = chunk.start_pts - chunk.pts_offset_ns

            pipeline.set_state(Gst.State.PAUSED)
            pipeline.get_state(Gst.CLOCK_TIME_NONE)

            pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT | Gst.SeekFlags.SNAP_BEFORE,
                start_pts,
            )

            # Set the pipeline to PLAYING and wait for end-of-stream or error
            pipeline.set_state(Gst.State.PLAYING)
            with TimeMeasure("Decode "):
                self._loop.run()
            pipeline.set_state(Gst.State.PAUSED)
            if old_pipeline:
                old_pipeline.set_state(Gst.State.NULL)

        if not retain_pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        # Return the cached raw preprocessed frames / jpegs and the corresponding timestamps.
        # Adjust for the PTS offset if any.
        self._cached_frames_pts = [t + chunk.pts_offset_ns / 1e9 for t in self._cached_frames_pts]

        # reset frame resoulution config after processing multiple files
        self._frame_width = self._frame_width_orig
        self._frame_height = self._frame_height_orig
        self._first_frame_width = 0
        self._first_frame_height = 0

        preprocessed_frames = self._preprocess(self._cached_frames)
        self._cached_frames = None
        self._frame_selector = frame_selector_backup

        return preprocessed_frames, self._cached_frames_pts

    def dispose_pipeline(self):
        if self._pipeline.set_state(Gst.State.NULL) != Gst.StateChangeReturn.SUCCESS:
            logger.error("Couldn't set state to NULL for pipeline")
        logger.info("Pipeline moved to NULL")

    def dispose_pipeline_from_separate_thread(self):
        """Safely move pipeline to NULL state and clean up resources."""

        # Create a flag to track completion
        self._disposal_complete = False

        def disposal_thread():
            """Thread function to handle pipeline disposal"""
            try:
                logger.debug("Starting pipeline disposal in separate thread")
                self.dispose_pipeline()
                self._disposal_complete = True
                logger.debug("Pipeline disposal completed")
            except Exception as e:
                logger.debug(f"Error during pipeline disposal: {e}")
                self._disposal_complete = True  # Mark as complete even on error

        # Start disposal thread
        disposal_thread = threading.Thread(target=disposal_thread)
        disposal_thread.start()

        # Wait for disposal to complete with timeout
        timeout = 120  # Total timeout in seconds
        start_time = time.time()
        while not self._disposal_complete:
            if time.time() - start_time > timeout:
                logger.error(f"ERROR: Pipeline disposal timed out after {timeout} seconds")
                break
            time.sleep(2)
            logger.debug("Waiting for pipeline disposal to complete...")

    def dispose_source(self, src):
        if src.set_state(Gst.State.NULL) != Gst.StateChangeReturn.SUCCESS:
            logger.error(f"Couldn't set state to NULL for {self._uridecodebin.get_name()}")
        logger.info("Source removed")

    def stream(
        self,
        live_stream_url: str,
        chunk_duration: int,
        on_chunk_decoded: Callable[[ChunkInfo, torch.Tensor | list[np.ndarray], list[float]], None],
        chunk_overlap_duration=0,
        username="",
        password="",
    ):
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        self._last_stream_id = ""

        self._live_stream_frame_selectors.clear()
        self._live_stream_url = live_stream_url
        self._live_stream_next_chunk_idx = 0
        self._live_stream_chunk_duration = chunk_duration
        self._live_stream_chunk_overlap_duration = chunk_overlap_duration
        self._live_stream_chunk_decoded_callback = on_chunk_decoded
        self._last_frame_pts = 0
        self._stop_stream = False

        # Rerun the pipeline if it runs into errors like disconnection
        # Stop if pipeline stops with EOS
        while ((not self._pipeline) or self._got_error) and (not self._stop_stream):
            if self._got_error:
                logger.error("Live stream received error. Retrying after 5 seconds")
                time.sleep(5)
            self._got_error = False
            self._live_stream_next_chunk_start_pts = 0
            self._live_stream_ntp_epoch = 0
            self._live_stream_ntp_pts = 0

            self._pipeline = self._create_pipeline(live_stream_url, username, password)
            logger.debug("Pipeline for live stream to PLAYING")
            self._pipeline.set_state(Gst.State.PLAYING)
            logger.debug("Pipeline for live stream to loop.run")
            self._loop.run()

            if self._rtspsrc:
                logger.debug(f"forcing EOS; {self._last_stream_id}")
                # Send EOS event to the source
                handled = self._rtspsrc.send_event(Gst.Event.new_eos())
                logger.debug(f"EOS forced; {handled} : {self._last_stream_id}")
                self._rtspsrc.set_property("timeout", 0)
                if self._udpsrc:
                    logger.debug(
                        f"forcing udpsrc timeout to 0 before teardown; {self._last_stream_id}"
                    )
                    self._udpsrc.set_property("timeout", 0)

            # Need to remove source bin and then move pipeline to NULL
            # to avoid Gst bug:
            # https://discourse.gstreamer.org/t/gstreamer-1-16-3-setting-rtsp-pipeline-to-null/538/11
            # TODO: Try latest GStreamer version for any fixes
            logger.debug(f"pipe teardown: unlink_source : {self._last_stream_id}")
            self._uridecodebin.unlink(self._q1)
            self._pipeline.remove(self._uridecodebin)

            # logger.debug(f"pipe teardown: to READY : {self._last_stream_id}")
            # self._pipeline.set_state(Gst.State.READY)
            # time.sleep(1)
            logger.debug(f"pipe teardown: to NULL : {self._last_stream_id}")
            self.dispose_pipeline_from_separate_thread()
            logger.debug(f"pipe teardown: dispose_source : {self._last_stream_id}")
            GLib.idle_add(self.dispose_source, self._rtspsrc)
            GLib.idle_add(self.dispose_source, self._uridecodebin)
            logger.debug(f"pipe teardown: done : {self._last_stream_id}")
            self._process_finished_chunks(flush=True)

        self._pipeline = None
        self._live_stream_frame_selectors.clear()

    def stop_stream(self):
        self._stop_stream = True
        logger.debug("Force quit loop")
        self._loop.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video File Frame Getter")
    parser.add_argument("file_or_rtsp", type=str, help="File / RTSP streams to frames from")

    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=10,
        help="Chunk duration in seconds to use for live streams",
    )
    parser.add_argument(
        "--chunk-overlap-duration",
        type=int,
        default=0,
        help="Chunk overlap duration in seconds to use for live streams",
    )
    parser.add_argument(
        "--username", type=str, default=None, help="Username to access the live stream"
    )
    parser.add_argument(
        "--password", type=str, default=None, help="Password to access the live stream"
    )

    parser.add_argument(
        "--start-time", type=int, default=0, help="Start time in sec to get frames from"
    )

    parser.add_argument(
        "--end-time", type=int, default=-1, help="End time in sec to get frames from"
    )

    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to get")

    parser.add_argument(
        "--enable-jpeg-output",
        type=bool,
        default=False,
        help="enable JPEG output instead of NVMM:x-raw",
    )

    args = parser.parse_args()

    frame_getter = VideoFileFrameGetter(
        frame_selector=DefaultFrameSelector(args.num_frames),
        enable_jpeg_output=args.enable_jpeg_output,
    )

    if args.file_or_rtsp.startswith("rtsp://"):
        frame_getter.stream(
            args.file_or_rtsp,
            chunk_duration=args.chunk_duration,
            chunk_overlap_duration=args.chunk_overlap_duration,
            username=args.username,
            password=args.password,
            on_chunk_decoded=lambda chunk, frames, frame_times: print(
                f"Picked {len(frames)} frames with times: {frame_times} for chunk {chunk}\n\n\n\n"
            ),
        )
    else:
        chunk = ChunkInfo()
        chunk.file = args.file_or_rtsp
        chunk.start_pts = args.start_time * 1000000000
        chunk.end_pts = args.end_time * 1000000000 if args.end_time >= 0 else -1
        frames, frames_pts = frame_getter.get_frames(chunk)
        print(f"Picked {len(frames)} frames with times: {frames_pts}")
