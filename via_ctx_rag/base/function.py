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

"""function.py: File contains Function class"""

from typing import Optional

from via_ctx_rag.base import Tool
from via_ctx_rag.utils.ctx_rag_logger import logger


class Function:
    """Function: This is a interface class that
    should be implemented to add a function to the Context Manager
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_setup: bool = False
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, Function] = {}
        self._params = {}

    def add_tool(self, name: str, tool: Tool):
        """Adds a tool to the function

        Args:
            name (str): Tool name
            tool (Tool): tool object

        Raises:
            RuntimeError: Raises error if another tool
            with same name already present
        """
        # TODO(sl): Try Catch with custom exception
        if name in self._tools:
            raise RuntimeError(f"Tool {name} already added in {self.name} function")
        self._tools[name] = tool
        return self

    def add_function(self, name: str, function: "Function"):
        """Adds a function to the current function's sub-function container.

        Args:
            name (str): The name of the function to add.
            function (Function): The function object to be added.

        Raises:
            RuntimeError: If a function with the same name is already added.
        """
        if name in self._functions:
            raise RuntimeError(f"Function {name} already added in {self.name} function")
        self._functions[name] = function
        return self

    def get_tool(self, name):
        return self._tools[name] if name in self._tools else None

    def get_function(self, name: str) -> Optional["Function"]:
        """Retrieve the sub-function associated with the given name.

        Args:
            name (str): The name of the function to retrieve.

        Returns:
            Optional[Function]: The function object if it exists; otherwise, None.
        """
        return self._functions[name] if name in self._functions else None

    async def __call__(self, state: dict) -> dict:
        if not self.is_setup:
            raise RuntimeError("Function not setup. Call done()!")
        result = await self.acall(state)
        return result

    async def aprocess_doc_(
        self, doc: str, doc_i: int, doc_meta: Optional[dict] = None
    ):
        if not self.is_setup:
            raise RuntimeError("Function not setup. Call done()!")
        if doc_meta is None:
            doc_meta = {}
        result = await self.aprocess_doc(doc, doc_i, doc_meta)
        return result

    def config(self, **params):
        logger.debug(f"Config params for {self.name}: {self._params} with {params}")
        self._params.update(params)
        logger.debug(f"Params configured for {self.name}: {self._params}")
        return self

    # TODO: Add def update(self) this will be added later.
    # We have to implement stop() which will ensure that
    # updating the config values is threadsafe
    def update(self, **params):
        self.config(**params)
        self.done()

    # function finds the value of a param from a nested dictionary
    # param is provided in the form of keys to traverse the dictionary
    # eg : to Obtain the batch_size for summarization, func.get_param("params", "batch_size")
    def get_param(self, *keys, required: bool = True, params: dict = None):
        if len(keys) == 0 and params is None:  # if no key is provided
            logger.info(f"======= PARAMS {params}")
            raise ValueError("Empty param provided.")
        if params is None:  # Top level function call before recursion begins
            params = self._params  # save an object reference to the param store
        if isinstance(params, dict):
            if len(keys) == 0:
                raise ValueError("Required more param keyss.")
            if keys[0] not in params:
                if required:  # key not found but required
                    raise ValueError(f"Required param {keys[0]} not configured.")
                else:  # key not found
                    return None
            else:  # Call the same function for traversing the inner dictionary obtained by indexing
                return self.get_param(
                    *keys[1:], required=required, params=params[keys[0]]
                )
        else:  # Reached the value in the dictionary
            if len(keys) == 0:  # there are no more keys provided to traverse,
                return params
            if len(keys) > 0:  # there are more keys provided to traverse
                raise ValueError(f"Required param {keys} not configured.")

    def done(self):
        self.setup()
        self.is_setup = True
        return self

    async def areset(self, expr):
        pass

    # TODO: change the function definition.
    # Pass **config, **tools, **functions instead of this
    # Or even better add _config, _tools and _functions in self and
    # expose a function like get_tool(), get_function(), get_config()
    def setup(self) -> dict:
        """This method is where the business logic of function
        should be implemented which can use tools. Each class
        extending Function class should implement this.
        """
        raise RuntimeError("`setup` method not Implemented!")

    async def acall(self, state: dict) -> dict:
        """This method is where the business logic of function
        should be implemented which can use tools. Each class
        extending Function class should implement this.

        Args:
            state (dict): This is the dict of the state
        """
        raise RuntimeError("`call` method not Implemented!")

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        """This method is called every time a doc is added to
        the Context Manager. The function has the option to process the
        doc when the doc is added.

        Args:
            doc (str): document
            i (int): document index
            meta (dict): document metadata
        """
        raise RuntimeError("`aprocess_doc` method not implemented")
