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

"""tool.py: File contains Tool class"""


class Tool:
    """Tool: This is a interface class that
    should be implemented to add a Tool that can be used in Functions
    """

    def __init__(self, name) -> None:
        self.name = name
