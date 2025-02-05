from pydantic import BaseModel, Field


from .playwright.proxy_launch_configs import ProxyLaunchConfigs
from .playwright.browser_launch_configs import BrowserLaunchConfigs


class Configs(BaseModel):
    pass






