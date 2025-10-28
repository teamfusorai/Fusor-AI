import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urljoin

import requests


def get_sitemap_urls(base_url: str, sitemap_filename: str = "sitemap.xml") -> List[str]:
    """Fetches and parses a sitemap XML file to extract URLs.

    Args:
        base_url: The base URL of the website
        sitemap_filename: The filename of the sitemap (default: sitemap.xml)

    Returns:
        List of URLs found in the sitemap. If sitemap is not found, returns a list
        containing only the base URL.

    Raises:
        ValueError: If there's an error fetching (except 404) or parsing the sitemap
    """
    try:
        sitemap_url = urljoin(base_url, sitemap_filename)

        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # Fetch sitemap URL
        response = requests.get(sitemap_url, timeout=15, headers=headers)

        # Handle different HTTP status codes gracefully
        if response.status_code == 404:
            print(f"Sitemap not found at {sitemap_url}, returning base URL")
            return [base_url.rstrip("/")]
        elif response.status_code == 403:
            print(f"Access forbidden for {sitemap_url}, returning base URL")
            return [base_url.rstrip("/")]
        elif response.status_code == 429:
            print(f"Rate limited for {sitemap_url}, returning base URL")
            return [base_url.rstrip("/")]

        response.raise_for_status()

        # Parse XML content
        root = ET.fromstring(response.content)

        # Handle different XML namespaces that sitemaps might use
        namespaces = (
            {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else ""
        )

        # Extract URLs using namespace if present
        if namespaces:
            urls = [elem.text for elem in root.findall(".//ns:loc", namespaces)]
        else:
            urls = [elem.text for elem in root.findall(".//loc")]

        # If no URLs found, return base URL
        if not urls:
            print(f"No URLs found in sitemap {sitemap_url}, returning base URL")
            return [base_url.rstrip("/")]

        return urls

    except requests.RequestException as e:
        print(f"Request failed for {sitemap_url}: {e}")
        return [base_url.rstrip("/")]
    except ET.ParseError as e:
        print(f"XML parsing failed for {sitemap_url}: {e}")
        return [base_url.rstrip("/")]
    except Exception as e:
        print(f"Unexpected error processing {sitemap_url}: {e}")
        return [base_url.rstrip("/")]


if __name__ == "__main__":
    print(get_sitemap_urls("https://ds4sd.github.io/docling/"))
