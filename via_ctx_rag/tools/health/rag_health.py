################################################################################
# Copyright (c) 5, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

import json


class GraphMetrics:
    def __init__(self):
        self.graph_create_tokens = 0
        self.graph_create_requests = 0
        self.graph_create_latency = 0

    def dump_json(self, file_name: str):
        """
        Dumps the object's attributes to a JSON file.

        Args:
            file_name (str, optional): The file name to write to.
        """
        data = {
            "graph_create_tokens": self.graph_create_tokens,
            "graph_create_requests": self.graph_create_requests,
            "graph_create_latency": self.graph_create_latency,
        }
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

    def reset(self):
        self.graph_create_tokens = 0
        self.graph_create_requests = 0
        self.graph_create_latency = 0


class SummaryMetrics:
    def __init__(self):
        self.summary_tokens = 0
        self.aggregation_tokens = 0
        self.summary_requests = 0
        self.summary_latency = 0
        self.aggregation_latency = 0

    def dump_json(self, file_name: str):
        """
        Dumps the object's attributes to a JSON file.

        Args:
            file_name (str, optional): The file name to write to.
        """
        data = {
            "summary_tokens": self.summary_tokens,
            "aggregation_tokens": self.aggregation_tokens,
            "summary_requests": self.summary_requests,
            "summary_latency": self.summary_latency,
            "aggregation_latency": self.aggregation_latency,
        }
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

    def reset(self):
        self.summary_tokens = 0
        self.aggregation_tokens = 0
        self.summary_requests = 0
        self.summary_latency = 0
        self.aggregation_latency = 0
