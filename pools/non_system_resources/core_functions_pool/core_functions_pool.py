


from typing import Any, Callable, Coroutine


from markitdown import MarkItDown
from pydantic import BaseModel


from utils.file_paths_manager.file_paths_manager import FilePathAndMetadata
from main import Resource


class FunctionAndParameters(BaseModel):
    specific_function: Callable | Coroutine
    args: dict[str, Any]
    kwargs: dict[str, Any]
    resource_estimate: dict[str, Any]


class FunctionDictionary(BaseModel):
    load_function: dict
    conversion_function: dict
    save_function: dict
    total_resource_estimate: dict[str, Any]
    total_resource_available: dict[str, Any]


class CoreFunctionsPool:
    """
    A management system for file format conversion operations that automatically selects and optimizes conversion strategies.

    This class provides an intelligent interface for converting files between different formats by:
    1. Determining if a conversion is possible
    2. Selecting the optimal conversion method based on historical performance
    3. Automatically tuning conversion parameters for best results
    4. Learning from successful conversions to improve future performance

    Key Features:
        - Automatic selection of optimal conversion strategy
        - Self-learning capability to improve conversion success rates
        - Resource usage optimization
        - Persistent storage of conversion knowledge
        
    Example Usage:
        >>> pool_manager = PoolResourceManager()
        >>> system_resource_pools = SystemResourcePools()
        >>> func_pool = CoreFunctionsPool("knowledge_base.json", max_memory_usage=2048)
        >>> resource = Resource("Path(path/to/document.docx")
        >>> can_convert, func_dict = func_pool.check_if_we_can_do_the_conversion(resource)
        >>> func_dict: FunctionDictionary
        >>> if can_convert:
        >>>     necessary_resources = pool_manager.see_what_system_resources_we_need_to_convert_this_file(resource, func_dict)
        >>>     resource.necessary_resources = necessary_resources
        >>>     resource.func_dict = func_dict

    Parameters:
        knowledge_base_path (str): Path to JSON file containing conversion knowledge.
        max_memory_usage (int, optional): Maximum memory usage in MB. Defaults to 1024.

    Methods:
        - check_if_we_can_do_the_conversion(resource: Resource) -> Optional[FunctionAndParameters]: 
            Checks to see if we can covert the file at all. 
            This is an orchestrator function that calls the other functions in this class.
            Returns a boolean of whether we can process the function,
                as well as the dictionary of the functions, args, and kwargs that did it.
            If false, then it only returns the boolean. 

        - _check_if_we_can_open_the_file(resource: Resource) -> tuple[bool, Optional[FunctionAndParameters]: 
                Check if file can be opened.
                This function checks to see if the file can be opened by any of the available libraries.
                Returns a boolean of whether the function can be opened, 
                    and a dictionary of function, args, and kwargs that opened the file.
                If false, then it only returns the boolean.

        - _check_if_we_can_convert_the_file(resource: Resource) -> tuple[bool, Optional[FunctionAndParameters]: 
            Check if file can be converted.
            This function checks to see if the file can be converted by any of the available libraries.
            Returns a boolean of whether the function can be converted, 
                   and a dictionary of function, args, and kwargs that converted the file.
            If false, then it only returns the boolean.

        - _check_if_we_can_save_the_file(resource: Resource) -> tuple[bool, Optional[FunctionAndParameters]: 
            Check if file can be saved.
            This function checks to see if the file can be converted by any of the available libraries.
            Returns a boolean of whether the function can be converted, 
                   and a dictionary of function, args, and kwargs that converted the file.
            If false, then it only returns the boolean.

        Check if file can be saved.

        - see_what_resources_we_need_to_convert_this_file(
                resource: Resource, func_dict: FunctionDictionary
            ) -> FunctionDictionary: 
                Get resource usage estimate.
                This is based on a number of different factors, 
                    including the disk size of the object, an estimate of how long it would take to convert,
                    and what resources are required by each function.
        
    Raises:
        FileNotFoundError: If knowledge base file cannot be found
        MemoryError: If conversion exceeds memory limits
        
    Notes:
        - Supports all MIME types in Markitdown library.
        - Conversion knowledge is persisted between runs via the knowledge_base.json
        - Thread-safe for concurrent conversions


    The CoreFunctionsPool class serves as a central repository of conversion functions and their metadata.
    At its core, it is a dictionary class that checks whether or not we can convert a file to a given format, 
        and how much resources it would take to do so.
    Given the large amount of MIME types that this needs to support, and the large amount of libraries that exist to do so,
        it is more practical to create mock inputs based on the file types and their size, and then
        iterate through a series of functions, as well as their arguments, to test which ones don't work.

    To this end, this class does the following:
        - The class provides methods to check if a file can be opened, converted, and saved.
        - It stores the functions, their args, and kwargs in a dictionary.
            It also stores the resources and time that each function takes to run, and their success rates.
            This is loaded from a JSON file.
        - Iterates through the dictionary, tries a function with the args and kwargs, to see if they produce an error.
        - If they produce an error, it changes one of the args and tries again.
            It does the same with kwargs.
            It does so until the args and kwargs are exhausted.
            If everything fails, we move onto the next function in the sequence.
        - If they succeed, it stores the function, args, and kwargs in the dictionary, as well as the resources and length of time it took to run.
            It also stores the specific values of the args and kwargs that were used.
            Then, it updates a JSON dictionary with the new values.
        
        - It also includes a method to iterate through the functions and find the most suitable one for a given file.
        - The class aims to optimize the conversion process by learning from previous attempts and storing successful 
            configurations.
        - It can handle various file formats and adapts to new file types by continuously updating its knowledge base.
        - The performance metrics stored for each function help in making informed decisions about which method to use 
            for different file types and sizes.
        - This approach allows for a flexible and extensible system that can accommodate new conversion 
            libraries and methods as they become available.
    """
    def __init__(self):
        pass

    def can_open(self, file_path: FilePathAndMetadata) -> bool:
        pass

    def can_convert(self, data: bytes, output_format: str) -> bool:
        pass

    def can_save(self, data: bytes, resource: Resource) -> bool:
        pass





















