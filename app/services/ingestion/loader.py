"""
Document loaders for multiple file formats and web URLs.
"""

import os
from typing import List, Optional
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Unified document loader for various formats."""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[str]:
        """
        Load PDF file and extract text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page texts
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            texts = [doc.page_content for doc in documents]
            logger.info(f"Loaded {len(texts)} pages from PDF: {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise
    
    @staticmethod
    def load_docx(file_path: str) -> List[str]:
        """
        Load DOCX file and extract text.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            List of paragraphs
        """
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            texts = [doc.page_content for doc in documents]
            logger.info(f"Loaded {len(texts)} sections from DOCX: {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path}: {e}")
            raise
    
    @staticmethod
    def load_txt(file_path: str) -> List[str]:
        """
        Load plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List containing full text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Loaded text file: {file_path}")
            return [text]
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            raise
    
    @staticmethod
    def load_excel(file_path: str) -> List[str]:
        """
        Load Excel file and extract text from all sheets.
        
        Args:
            file_path: Path to Excel file (.xlsx, .xls)
            
        Returns:
            List of sheet texts
        """
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            texts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert dataframe to text representation
                sheet_text = f"Sheet: {sheet_name}\n\n"
                
                # Add column headers
                sheet_text += " | ".join(str(col) for col in df.columns) + "\n"
                sheet_text += "-" * 80 + "\n"
                
                # Add rows
                for _, row in df.iterrows():
                    row_text = " | ".join(str(val) for val in row.values)
                    sheet_text += row_text + "\n"
                
                texts.append(sheet_text)
            
            logger.info(f"Loaded {len(texts)} sheets from Excel: {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Failed to load Excel {file_path}: {e}")
            raise
    
    @staticmethod
    def load_csv(file_path: str) -> List[str]:
        """
        Load CSV file and extract text.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List containing CSV text
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Convert to text representation
            text = "CSV Data\n\n"
            
            # Add column headers
            text += " | ".join(str(col) for col in df.columns) + "\n"
            text += "-" * 80 + "\n"
            
            # Add rows
            for _, row in df.iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text += row_text + "\n"
            
            logger.info(f"Loaded CSV file: {file_path}")
            return [text]
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise
    
    @staticmethod
    def _extract_main_content(soup: BeautifulSoup, url: str) -> str:
        """
        Extract main article content from a page, removing boilerplate.
        Safe: collects elements to remove BEFORE removing them (avoids BS4 tree mutation bug).
        """
        # ── Step 1: grab title & meta BEFORE we remove anything ──────────
        title_tag = soup.find("title")
        meta_tag  = soup.find("meta", attrs={"name": "description"})

        title_text = title_tag.get_text(strip=True) if title_tag else ""
        meta_text  = ""
        if meta_tag and isinstance(meta_tag.attrs, dict):
            meta_text = (meta_tag.attrs.get("content") or "").strip()

        header_parts = []
        if title_text:
            header_parts.append(f"Title: {title_text}")
        if meta_text:
            header_parts.append(f"Description: {meta_text}")
        header_parts.append(f"Source: {url}")
        header = "\n".join(header_parts) + "\n\n"

        # ── Step 2: collect tags to strip, THEN remove them ──────────────
        structural_tags = {"script", "style", "noscript", "iframe",
                           "nav", "header", "footer", "aside",
                           "form", "button", "input", "select",
                           "svg", "img", "figure", "picture", "video", "audio"}

        boilerplate_patterns = [
            "cookie", "banner", "popup", "modal", "overlay",
            "advertisement", "ad-", "ads-", "sidebar", "widget",
            "newsletter", "subscribe", "social-share", "breadcrumb",
            "pagination", "site-menu", "navbar", "topbar", "bottombar",
            "skip-link", "skip-to", "visually-hidden"
        ]

        tags_to_remove = []
        for tag in soup.find_all(True):          # collect first
            try:
                if tag.name in structural_tags:
                    tags_to_remove.append(tag)
                    continue
                tag_id    = (tag.get("id") or "").lower()
                tag_class = " ".join(tag.get("class") or []).lower()
                combined  = tag_id + " " + tag_class
                if any(p in combined for p in boilerplate_patterns):
                    tags_to_remove.append(tag)
            except Exception:
                continue

        for tag in tags_to_remove:               # remove AFTER iterating
            try:
                tag.decompose()
            except Exception:
                pass

        # ── Step 3: find best content container ──────────────────────────
        def _has_text(el):
            return el is not None and len(el.get_text(strip=True)) > 50

        candidates = [
            soup.find("article"),
            soup.find("main"),
            soup.find(attrs={"role": "main"}),
            soup.find("div", id="content"),
            soup.find("div", id="main-content"),
            soup.find("div", id="bodyContent"),      # Wikipedia
            soup.find("div", class_="post-content"),
            soup.find("div", class_="article-body"),
            soup.find("div", class_="entry-content"),
            soup.find("body"),
        ]

        main_content = soup  # ultimate fallback
        for candidate in candidates:
            if _has_text(candidate):
                main_content = candidate
                break

        # ── Step 4: extract, filter prose, and clean text ────────────────
        raw = main_content.get_text(separator="\n")

        # Filter for prose-quality lines:
        # – drop very short lines (nav items, bullets, menu entries, lone words)
        # – a real sentence usually has >= 6 words and ends with punctuation
        #   OR is at least 60 characters (handles long mid-sentence fragments)
        import re as _re

        MIN_LINE_CHARS = 40   # minimum characters for a line to be kept
        MIN_LINE_WORDS = 6    # minimum words for a line to be kept

        cleaned = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                cleaned.append("")    # preserve blank line for paragraph spacing
                continue
            word_count = len(line.split())
            # Keep if long enough OR has sentence-ending punctuation and >= 6 words
            if len(line) >= MIN_LINE_CHARS or (word_count >= MIN_LINE_WORDS and line[-1] in ".?!:;\"'"):
                cleaned.append(line)
            # else: silently drop nav fragments, bullets, single words, etc.

        # Collapse multiple consecutive blank lines to a single gap (keep \n\n paragraphs)
        collapsed = []
        blank_count = 0
        for line in cleaned:
            if not line:
                blank_count += 1
                if blank_count <= 1:
                    collapsed.append("")
            else:
                blank_count = 0
                collapsed.append(line)

        body_text = "\n".join(collapsed).strip()

        if not body_text:
            # Ultimate fallback: raw page text (unfiltered)
            body_text = soup.get_text(separator="\n", strip=True)

        return header + body_text

    # Full browser-like headers to avoid bot-blocking
    _BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    @staticmethod
    def _fetch_wikipedia(url: str) -> Optional[str]:
        """
        Fetch a Wikipedia article using multiple strategies (most reliable first).

        Strategy 1 – MediaWiki action API (w/api.php, plain-text extract, JSON):
            Never blocked; returns clean prose ready to use.
        Strategy 2 – REST v1 HTML endpoint:
            Usually works but occasionally returns 403.
        Strategy 3 – Mobile site (en.m.wikipedia.org):
            Much less aggressive bot-blocking than the desktop site.

        Works for any en.wikipedia.org/wiki/Article_Name URL.
        """
        import re as _re
        import httpx

        match = _re.search(r"wikipedia\.org/wiki/(.+?)(?:#.*)?$", url)
        if not match:
            return None
        title = match.group(1)  # URL-encoded title, e.g. Healthcare_in_the_United_States

        # ── Strategy 1: MediaWiki action API ─────────────────────────────────
        # Returns clean, plain-text article content as JSON.  No rate-limit for
        # anonymous reads; this is the officially supported programmatic path.
        try:
            api_url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query"
                f"&titles={title}"
                f"&prop=extracts"
                f"&explaintext=true"   # plain text, no wikitext markup
                f"&exsectionformat=plain"
                f"&format=json"
                f"&utf8=1"
            )
            resp = httpx.get(
                api_url,
                timeout=30,
                headers={"User-Agent": "LLM-Data-Factory/1.0 (research project; contact: admin@example.com)"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    logger.warning(f"Wikipedia article not found: {title}")
                    break
                extract = page_data.get("extract", "").strip()
                if len(extract) > 200:
                    page_title = page_data.get("title", title)
                    full_text = f"Title: {page_title}\nSource: {url}\n\n{extract}"
                    logger.info(
                        f"Wikipedia MediaWiki API succeeded: {len(full_text)} chars for '{page_title}'"
                    )
                    return full_text
        except Exception as e:
            logger.warning(f"Wikipedia MediaWiki API strategy failed: {e}")

        # ── Strategy 2: REST v1 HTML endpoint ─────────────────────────────────
        try:
            rest_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{title}"
            resp = httpx.get(
                rest_url,
                timeout=30,
                headers=DocumentLoader._BROWSER_HEADERS,
                follow_redirects=True,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = DocumentLoader._extract_main_content(soup, url)
            if len(text.strip()) > 200:
                logger.info(f"Wikipedia REST v1 HTML strategy succeeded: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"Wikipedia REST v1 HTML strategy failed: {e}")

        # ── Strategy 3: Mobile site ───────────────────────────────────────────
        try:
            mobile_url = f"https://en.m.wikipedia.org/wiki/{title}"
            resp = httpx.get(
                mobile_url,
                timeout=30,
                headers=DocumentLoader._BROWSER_HEADERS,
                follow_redirects=True,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = DocumentLoader._extract_main_content(soup, url)
            if len(text.strip()) > 200:
                logger.info(f"Wikipedia mobile-site strategy succeeded: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"Wikipedia mobile-site strategy failed: {e}")

        return None

    # Keep old name as an alias so any external callers are unaffected
    @staticmethod
    def _fetch_wikipedia_api(url: str) -> Optional[str]:
        return DocumentLoader._fetch_wikipedia(url)

    @staticmethod
    def load_url(url: str, max_depth: int = 1) -> List[str]:
        """
        Load content from URL with smart content extraction.
        Uses full browser headers with retry logic and a Wikipedia API fallback.

        Args:
            url: Target URL
            max_depth: Crawl depth (1 = single page, 2 = follow links)

        Returns:
            List of page texts (cleaned, boilerplate-free)
        """
        import time as _time
        import httpx

        MAX_RETRIES = 3
        RETRY_DELAY = 2  # seconds

        if max_depth == 1:
            text = None

            # ── Fast-path for Wikipedia: skip browser-fetch attempts that are
            #    almost always blocked (403).  Go straight to the reliable API.
            if "wikipedia.org" in url:
                logger.info(f"Wikipedia URL detected – using API strategies directly: {url}")
                wiki_text = DocumentLoader._fetch_wikipedia(url)
                if wiki_text:
                    return [wiki_text]
                raise ValueError(
                    f"All Wikipedia fetch strategies failed for: {url}. "
                    f"The article may not exist or the Wikipedia API is temporarily unavailable."
                )

            # --- Attempt 1-3: direct fetch with full browser headers ---
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(f"URL fetch attempt {attempt}/{MAX_RETRIES}: {url}")
                    response = httpx.get(
                        url,
                        timeout=30,
                        headers=DocumentLoader._BROWSER_HEADERS,
                        follow_redirects=True,
                    )
                    response.raise_for_status()

                    soup = BeautifulSoup(response.text, "html.parser")
                    text = DocumentLoader._extract_main_content(soup, url)

                    if len(text.strip()) >= 200:
                        logger.info(f"Loaded URL on attempt {attempt} ({len(text)} chars): {url}")
                        return [text]

                    logger.warning(
                        f"Attempt {attempt} returned too little content "
                        f"({len(text.strip())} chars). Retrying..."
                    )
                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed: {e}")

                if attempt < MAX_RETRIES:
                    _time.sleep(RETRY_DELAY)

            # --- All attempts exhausted ---
            preview = (text or "")[:200]
            raise ValueError(
                f"Failed to crawl URL after {MAX_RETRIES} attempts. "
                f"The site may be blocking automated requests. "
                f"Content preview: {preview}"
            )

        else:
            # Multi-page recursive crawling with smart extraction per page
            try:
                loader = RecursiveUrlLoader(
                    url=url,
                    max_depth=max_depth,
                    extractor=lambda html: DocumentLoader._extract_main_content(
                        BeautifulSoup(html, "html.parser"), url
                    ),
                )
                documents = loader.load()
                texts = [doc.page_content for doc in documents if doc.page_content.strip()]
                logger.info(f"Loaded {len(texts)} pages from URL (smart extract): {url}")
                return texts
            except Exception as e:
                logger.error(f"Failed to load URL {url}: {e}")
                raise
    
    @staticmethod
    def load_file(file_path: str) -> List[str]:
        """
        Auto-detect file type and load accordingly.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of text segments
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return DocumentLoader.load_docx(file_path)
        elif ext in ['.txt', '.md']:
            return DocumentLoader.load_txt(file_path)
        elif ext in ['.xlsx', '.xls']:
            return DocumentLoader.load_excel(file_path)
        elif ext == '.csv':
            return DocumentLoader.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
