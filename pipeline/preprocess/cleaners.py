"""Text cleaning, HTML stripping, and encoding fixes for preprocessing."""

import re
import ftfy
from bs4 import BeautifulSoup

_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")

def fix_encoding(text: str) -> str:
    """Automatically fix broken unicode encodings (mojibake)."""
    if not text:
        return text
    return ftfy.fix_text(text)

def strip_html(html_content: str) -> str:
    """Remove HTML tags, scripts, styles, and extract clean text."""
    if not html_content:
        return html_content
    
    # Use BeautifulSoup to parse
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        script_or_style.decompose()
        
    # Get text
    text = soup.get_text(separator="\n")
    return text

def clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace and limit consecutive blank lines."""
    if not text:
        return text
    cleaned = _WHITESPACE.sub(" ", text)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()

def remove_ocr_garbage(text: str) -> str:
    """Basic heuristics to remove garbled OCR lines.
    For example, lines with too many special characters vs normal characters.
    """
    if not text:
        return text
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        if not line.strip():
            cleaned_lines.append(line)
            continue
            
        # If line is more than 50% non-alphanumeric (excluding spaces), it might be garbage
        alnum_count = sum(c.isalnum() for c in line)
        non_space_count = len(line.replace(" ", ""))
        
        if non_space_count > 0:
            ratio = alnum_count / non_space_count
            if ratio < 0.5:
                # Likely garbage, skip this line
                continue
                
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def process_text(raw_text: str, is_html: bool = False) -> str:
    """Run all cleaning steps."""
    text = raw_text
    if is_html:
        text = strip_html(text)
    
    text = fix_encoding(text)
    text = remove_ocr_garbage(text)
    text = clean_whitespace(text)
    
    return text
