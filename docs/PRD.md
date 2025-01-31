

# Program Name: convert_to_txt_based_on_mime_type
# Mark 1 :0.1.0
# Author: Kyle Rose, Claude 3.5 Sonnet, Codestral

# Description: Convert any kind of file from the web into a txt document based on its MIME type.

# Long Description: This module attempts to convert files from specified URLs into text files based on their MIME types. It is designed to handle common file formats such as PDF, DOCX, and HTML, with modularity in mind to allow for easy expansion to future formats.


## Overview

## NOTE
This module will not work, or is meant to be, a standalone program. It is intended to be imported and used as a utility for other projects.

## Key Features
- Dynamic MIME type detection and processing
- Automated format conversion to plain text
- Flexible URL input handling
- Robust error management for failed conversions

Dependencies
- Markitdown
- Playwright
- aiohttp
- pydantic