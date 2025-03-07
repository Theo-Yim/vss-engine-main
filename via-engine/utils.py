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
"""Common utility methods."""

import asyncio
import json
import os
import re
import subprocess

import gi
import yaml
from pymediainfo import MediaInfo

# from json_minify import json_minify

gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")

from gi.repository import Gst, GstPbutils  # noqa: E402

Gst.init(None)


class MediaFileInfo:
    is_image = False
    video_codec = ""
    video_duration_nsec = 0
    video_fps = 0.0
    video_resolution = (0, 0)

    @staticmethod
    def _get_info_gst(uri_or_file: str, username="", password=""):
        uri_or_file = str(uri_or_file)
        media_file_info = MediaFileInfo()

        if uri_or_file.startswith("rtsp://") or uri_or_file.startswith("file://"):
            uri = uri_or_file
        else:
            uri = "file://" + os.path.abspath(str(uri_or_file))

        def select_stream(source, idx, caps):
            if "audio" in caps.to_string():
                return False
            return True

        def source_setup(discoverer, source):
            if uri.startswith("rtsp://"):
                source.connect("select-stream", select_stream)
                source.set_property("timeout", 1000000)
                if username and password:
                    source.set_property("user-id", username)
                    source.set_property("user-pw", password)

        discoverer = GstPbutils.Discoverer()
        discoverer.connect("source-setup", source_setup)

        try:
            file_info = discoverer.discover_uri(uri)
        except gi.repository.GLib.GError as e:
            raise Exception("Unsupported file type - " + uri + " Error:" + str(e))
        for stream_info in file_info.get_stream_list():
            if isinstance(stream_info, GstPbutils.DiscovererVideoInfo):
                media_file_info.video_duration_nsec = int(file_info.get_duration())
                media_file_info.video_codec = str(
                    GstPbutils.pb_utils_get_codec_description(stream_info.get_caps())
                )
                media_file_info.video_resolution = (
                    int(stream_info.get_width()),
                    int(stream_info.get_height()),
                )
                media_file_info.video_fps = float(
                    stream_info.get_framerate_num() / stream_info.get_framerate_denom()
                )
                media_file_info.is_image = bool(stream_info.is_image())
                break
        return media_file_info

    @staticmethod
    def _get_info_mediainfo(uri_or_file: str):
        if uri_or_file.startswith("file://"):
            file = uri_or_file[7:]
        else:
            file = uri_or_file

        media_file_info = MediaFileInfo()
        media_info = MediaInfo.parse(file)
        have_image_or_video = False
        for track in media_info.tracks:
            if track.track_type == "Video":
                media_file_info.is_image = False
                media_file_info.video_codec = track.format
                media_file_info.video_duration_nsec = track.duration * 1000000
                media_file_info.video_fps = track.frame_rate
                media_file_info.video_resolution = (track.width, track.height)
                have_image_or_video = True
            if track.track_type == "Image":
                media_file_info.is_image = True
                media_file_info.video_codec = track.format
                media_file_info.video_duration_nsec = 0
                media_file_info.video_fps = 0
                media_file_info.video_resolution = (track.width, track.height)
                have_image_or_video = True

        if not have_image_or_video:
            raise Exception("Unsupported file type - " + file)
        return media_file_info

    @staticmethod
    def get_info(uri_or_file: str, username="", password=""):
        if str(uri_or_file).startswith("rtsp://"):
            return MediaFileInfo._get_info_gst(uri_or_file, username, password)
        else:
            return MediaFileInfo._get_info_mediainfo(str(uri_or_file))

    @staticmethod
    async def get_info_async(uri_or_file: str, username="", password=""):
        return await asyncio.get_event_loop().run_in_executor(
            None, MediaFileInfo.get_info, uri_or_file, username, password
        )


def round_up(s):
    """
    Rounds up a string representation of a number to an integer.

    Example:
    >>> round_up("7.9s")
    8
    """
    # Strip any non-numeric characters from the string
    num_str = re.sub(r"[a-zA-Z]+", "", s)

    # Convert the string to a float and round up to the nearest integer
    num = float(num_str)
    return -(-num // 1)  # equivalent to math.ceil(num) in Python 3.x


def get_avg_time_per_chunk(GPU_in_use, Model_ID, yaml_file_path):
    """
    Returns the average time per query for a given GPU and Model ID
    from a VIA_runtime_stats YAML file.

    Args:
        GPU_in_use (str): The GPU in use (e.g. A100, H100)
        Model_ID (str): The Model ID (e.g. VILA)
        yaml_file_path (str): The path to the VIA_runtime_stats YAML file

    Returns:
        str: The average time per chunk (e.g. 2.5s, 1.8s)
    """

    def is_subset_s1_in_s2(string1, string2):
        # Returns True if string1 is a subset of string2, ignoring case
        pattern = re.compile(re.escape(string1), re.IGNORECASE)
        return bool(pattern.search(string2))

    def is_subset(string1, string2):
        return is_subset_s1_in_s2(string1, string2) or is_subset_s1_in_s2(string2, string1)

    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    max_atpc = 0.0
    max_atpc_as_is = "0"

    for entry in yaml_data["VIA_runtime_stats"]:
        if round_up(entry["average_time_per_chunk"]) > max_atpc:
            max_atpc = round_up(entry["average_time_per_chunk"])
            max_atpc_as_is = entry["average_time_per_chunk"]
        if is_subset(GPU_in_use, entry["GPU_in_use"]) and is_subset(Model_ID, entry["Model_ID"]):
            return entry["average_time_per_chunk"]

    # If no matching entry is found, return max of all
    return max_atpc_as_is


def get_available_gpus():
    """
    Returns an array of available NVIDIA GPUs with their names and memory sizes.

    Example output:
    [
        {"name": "GeForce RTX 3080", "memory": "12288 MiB"},
        {"name": "Quadro RTX 4000", "memory": "16384 MiB"}
    ]
    """
    try:
        # Run nvidia-smi command and capture output
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        )

        # Split output into lines
        lines = output.decode("utf-8").strip().split("\n")

        # Initialize empty list to store GPU info
        gpus = []

        # Iterate over lines and extract GPU info
        for line in lines:
            cols = line.split(",")
            gpu_name = cols[0].strip()
            gpu_memory = cols[1].strip()
            gpus.append({"name": gpu_name, "memory": gpu_memory})

        return gpus

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []


# Convert the matrix to bit strings
def matrix_to_bit_strings(matrix):
    return ["".join(map(str, row)) for row in matrix]


# Run-length encoding function
def rle_encode(matrix):
    encoded = []
    for row in matrix:
        row_encoded = []
        current_value = row[0]
        count = 1
        for i in range(1, len(row)):
            if row[i] == current_value:
                count += 1
            else:
                row_encoded.append((int(current_value), count))
                current_value = row[i]
                count = 1
        row_encoded.append((int(current_value), count))
        encoded.append(row_encoded)
    return encoded


def find_object_with_key_value(json_array, target_key, target_value):
    # Loop through each object in the JSON array (list of dicts)
    for obj in json_array:
        if isinstance(obj, dict):
            # Check if the key-value pair exists in the current object
            if obj.get(target_key) == target_value:
                return obj  # Return the entire object if a match is found
    return None  # Return None if no match is found


class JsonCVMetadata:
    def __init__(self, request_id):
        self.data = []  # Initialize an empty list to store entries
        self._request_id = request_id

    def write_frame(self, frame_meta):
        import pyds

        # Frame level metadata
        frame = {}
        # frame["requestId"] = self._request_id
        # frame["version"] = "abc"
        frame["frameNo"] = frame_meta.frame_num
        # frame["fileName"] = "abc"
        frame["timestamp"] = frame_meta.buf_pts
        # frame["sensorId"] = frame_meta.source_id
        # frame["model"] = "VIA_CV"
        frame["frameWidth"] = frame_meta.source_frame_width
        frame["frameHeight"] = frame_meta.source_frame_height
        # frame["grounding"] = {}
        # Object level metadata
        objects = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            object = {}
            object["id"] = obj_meta.object_id
            bbox = {}
            bbox["lX"] = round(obj_meta.rect_params.left, 2)
            bbox["tY"] = round(obj_meta.rect_params.top, 2)
            bbox["rX"] = round((obj_meta.rect_params.left + obj_meta.rect_params.width), 2)
            bbox["bY"] = round((obj_meta.rect_params.top + obj_meta.rect_params.height), 2)
            object["bbox"] = bbox
            object["type"] = obj_meta.obj_label
            object["conf"] = round(obj_meta.confidence, 2)
            # misc metadata for each object
            misc = []
            misc_object = {}
            misc_object["chId"] = -1
            misc_object["bbox"] = bbox
            misc_object["conf"] = round(obj_meta.confidence, 2)
            # mask data for each object
            segmentation = {}
            # mask = obj_meta.mask_params.get_mask_array().reshape(
            #    obj_meta.mask_params.height, obj_meta.mask_params.width
            # )
            # mask_uint8 = mask.astype(numpy.uint8)
            # bit_strings = matrix_to_bit_strings(mask_uint8)
            # encoded_matrix = rle_encode(mask_uint8)
            # segmentation["mask"] = encoded_matrix #bit_strings #mask_uint8.tolist()
            misc_object["seg"] = segmentation
            misc.append(misc_object)
            object["misc"] = misc
            objects.append(object)
            l_obj = l_obj.next
        frame["objects"] = objects
        self.data.append(frame)

    def write_past_frame_meta(self, batch_meta):
        import pyds

        l_user = batch_meta.batch_user_meta_list
        pastFrameObjList = {}
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if (
                user_meta
                and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META
            ):
                try:
                    pPastFrameObjBatch = pyds.NvDsTargetMiscDataBatch.cast(
                        user_meta.user_meta_data
                    )  # See NvDsTargetMiscDataBatch for details
                except StopIteration:
                    break
                for trackobj in pyds.NvDsTargetMiscDataBatch.list(
                    pPastFrameObjBatch
                ):  # Iterate through list of NvDsTargetMiscDataStream objects
                    for pastframeobj in pyds.NvDsTargetMiscDataStream.list(
                        trackobj
                    ):  # Iterate through list of NvDsFrameObjList objects
                        # numobj = pastframeobj.numObj
                        uniqueId = pastframeobj.uniqueId
                        # classId = pastframeobj.classId
                        objLabel = pastframeobj.objLabel
                        for objlist in pyds.NvDsTargetMiscDataObject.list(
                            pastframeobj
                        ):  # Iterate through list of NvDsFrameObj objects
                            bbox = {}
                            bbox["lX"] = round(objlist.tBbox.left, 2)
                            bbox["tY"] = round(objlist.tBbox.top, 2)
                            bbox["rX"] = round((objlist.tBbox.left + objlist.tBbox.width), 2)
                            bbox["bY"] = round((objlist.tBbox.top + objlist.tBbox.height), 2)
                            bbox["conf"] = round(objlist.confidence, 2)
                            bbox["id"] = uniqueId
                            bbox["type"] = objLabel
                            frameNum = objlist.frameNum
                            if frameNum not in pastFrameObjList:
                                pastFrameObjList[frameNum] = []
                            pastFrameObjList[frameNum].append(bbox)
            try:
                l_user = l_user.next
            except StopIteration:
                break

        # Now that pastFrameObjList is filled, add it to json metadata
        for frameNum, pastFrameObjects in pastFrameObjList.items():
            frameObject = find_object_with_key_value(self.data, "frameNo", frameNum)
            if frameObject:
                for pastObject in pastFrameObjects:
                    object = {}
                    object["id"] = pastObject["id"]
                    bbox = {}
                    bbox["lX"] = pastObject["lX"]
                    bbox["tY"] = pastObject["tY"]
                    bbox["rX"] = pastObject["rX"]
                    bbox["bY"] = pastObject["bY"]
                    object["bbox"] = bbox
                    object["type"] = pastObject["type"]
                    object["conf"] = pastObject["conf"]
                    # misc metadata for each object
                    misc = []
                    misc_object = {}
                    misc_object["chId"] = -1
                    misc_object["bbox"] = bbox
                    misc_object["conf"] = pastObject["conf"]
                    misc.append(misc_object)
                    object["misc"] = misc
                    frameObject["objects"].append(object)
            else:
                print(
                    f"write_past_frame_meta:Couldn't find json object with frame number={frameNum}"
                )
                print("Ignoring the data")

    def write_json_file(self, filename: str):
        with open(filename, "w") as file:
            json.dump(self.data, file, separators=(",", ":"))
        #    json.dump(self.data, file, indent=4)
        # json_string = json.dumps(self.data)
        # minified_json = json_minify(json_string)
        # with open(filename, 'w') as file:
        #    file.write(minified_json)

    def get_pts_to_frame_num_map(self):
        pts_to_frame_num_map = {obj["timestamp"]: obj["frameNo"] for obj in self.data}
        # print(pts_to_frame_num_map)
        return pts_to_frame_num_map


def get_json_file_name(request_id, chunk_idx):
    filename = str(request_id) + "_" + str(chunk_idx) + ".json"
    return filename
