
from typing import Any


from pydantic import BaseModel, Field
from pydantic_models.configs import Configs



class ExternalResourceManager():


    def __init__(self, file_paths_and_metadata, configs: Configs):

        self.configs = configs
        self.file_paths_and_metadata = file_paths_and_metadata









