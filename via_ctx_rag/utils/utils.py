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

"""utils.py: File contains utility functions"""

import jsonschema
import json
import re
from via_ctx_rag.utils.ctx_rag_logger import logger


def validate_config(parsed_yaml, schema_json_filepath):
    try:
        with open(schema_json_filepath) as f:
            spec_schema = json.load(f)
        jsonschema.validate(parsed_yaml, spec_schema)
    except jsonschema.ValidationError as e:
        raise ValueError(
            f"Invalid config file: {'.'.join([str(p) for p in e.absolute_path])}: {e.message}"
        )


def remove_think_tags(text_in):
    logger.info("Model Output: %s", text_in)
    text_out = re.sub(r"<think>.*?</think>", "", text_in, flags=re.DOTALL)
    logger.debug("Filtered Output: %s", text_out)
    return text_out


def remove_lucene_chars(text: str) -> str:
    """
    Remove Lucene special characters from the given text.

    This function takes a string as input and removes any special characters
    that are used in Lucene query syntax. The characters removed are:
    +, -, &, |, !, (, ), {, }, [, ], ^, ", ~, *, ?, :, \ and /.

    Args:
        text (str): The input string from which to remove Lucene special characters.

    Returns:
        str: The cleaned string with Lucene special characters replaced by spaces.
    """
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
        "/",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()
