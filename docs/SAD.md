



# Program Flow

1. Load in files to be converted.
    - If mode is "files" then: Load in a list of files from a folder.
    - If mode is "urls" then:
    - Load in a list of URLs from a file.
    - Supported file types: JSON, CSV, TXT.
    2. For each URL, perform the following steps:
    - Check if the URL is valid.
    - If not valid, log it and skip to the next URL.
            - Valid URL: A URL that starts with "http://" or "https://" and has a valid domain name.
    - If valid, download the content of the URL.
2. For each file, perform the following steps: