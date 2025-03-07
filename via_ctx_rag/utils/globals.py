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

CONFIG_SCHEMA = "data/config_schema.json"
DEFAULT_GRAPH_RAG_BATCH_SIZE = 1
DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE = 5
DEFAULT_RAG_TOP_K = 5
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_LLM_PARAMS = {
    "model": "meta/llama3-70b-instruct",
    "max_tokens": 1024,
    "top_p": 1,
    "temperature": 0.4,
    "seed": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
DEFAULT_SUMM_RECURSION_LIMIT = 8
