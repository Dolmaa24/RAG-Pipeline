"""Email: headers, body, and every attachment as a child item.

The attachments are the point. An ``.eml`` is usually a two-line note with the
actual content in a PDF hanging off it, so an email handler that reads only the
body reads the wrong thing. Each attachment becomes a child, goes back through
the type router, and is handled as whatever it turns out to be — including
another email.
"""

from __future__ import annotations

import email
import email.policy
from email.message import EmailMessage
from typing import Optional

from config import config
from errors import MissingDependency, ParseError
from models import ExtractionItem, ResourceKind
from observability import get_logger

from .base import BaseHandler, registry

log = get_logger("handlers.email")

#: Headers worth keeping. The full set on a delivered message runs to dozens of
#: Received: lines that say nothing about the content.
_KEEP_HEADERS = (
    "from", "to", "cc", "bcc", "reply-to", "subject", "date",
    "message-id", "in-reply-to", "references", "list-id",
)

MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024


class EmailHandler(BaseHandler):
    name = "email"
    kinds = (ResourceKind.EMAIL,)

    def process(self, item: ExtractionItem) -> ExtractionItem:
        if not item.raw_bytes:
            raise ParseError("no message bytes")

        if item.metadata.get("detected_subtype") == "msg":
            return self._process_outlook(item)

        try:
            message = email.message_from_bytes(item.raw_bytes, policy=email.policy.default)
        except Exception as exc:
            raise ParseError(f"could not parse message: {exc}") from exc

        headers = {
            name: str(message[name]) for name in _KEEP_HEADERS if message[name] is not None
        }
        item.metadata.update(headers)
        item.metadata["title"] = headers.get("subject") or "(no subject)"

        body = self._body(message)
        attachments = self._attachments(item, message)

        item.structured = {"email": {"headers": headers, "attachments": attachments}}
        item.cleaned_text = self._render(headers, body, attachments)
        log.info(
            "email.parsed",
            url=item.url,
            subject=headers.get("subject", "")[:80],
            attachments=len(attachments),
        )
        return item

    # ------------------------------------------------------------------ #

    @staticmethod
    def _body(message: EmailMessage) -> str:
        """Prefer text/plain; fall back to stripping the HTML alternative."""
        try:
            part = message.get_body(preferencelist=("plain", "html"))
        except Exception:
            part = None
        if part is None:
            return ""

        try:
            content = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True) or b""
            content = payload.decode(part.get_content_charset() or "utf-8", "replace")

        if part.get_content_type() == "text/html":
            try:
                from selectolax.lexbor import LexborHTMLParser

                return LexborHTMLParser(content).text(separator="\n", strip=True)
            except Exception:
                pass
        return content.strip()

    def _attachments(self, item: ExtractionItem, message: EmailMessage) -> list[dict]:
        found: list[dict] = []
        for part in message.iter_attachments():
            filename = part.get_filename() or "attachment"
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            record = {
                "filename": filename,
                "content_type": part.get_content_type(),
                "bytes": len(payload),
            }
            found.append(record)

            if len(payload) > MAX_ATTACHMENT_BYTES:
                item.warn(f"attachment {filename} skipped: {len(payload)} bytes")
                continue
            if item.depth >= config.MAX_RECURSION_DEPTH:
                continue

            child = ExtractionItem(
                url=f"{item.url}!/{filename}",
                depth=item.depth + 1,
                parent_url=item.url,
                raw_bytes=payload,
                content_type=part.get_content_type(),
                metadata={"attachment_of": item.url, "filename": filename},
            )
            child.compute_content_hash()
            item.children.append(child)
        return found

    @staticmethod
    def _render(headers: dict, body: str, attachments: list[dict]) -> str:
        lines = [f"{name.title()}: {value}" for name, value in headers.items()]
        if attachments:
            names = ", ".join(a["filename"] for a in attachments)
            lines.append(f"Attachments: {names}")
        lines.extend(["", body])
        return "\n".join(lines).strip()

    @staticmethod
    def _process_outlook(item: ExtractionItem) -> ExtractionItem:
        """Outlook .msg is an OLE2 compound file, not RFC 822."""
        try:
            import extract_msg
        except ImportError as exc:
            raise MissingDependency("extract-msg", "reading Outlook .msg files") from exc

        import io

        message = extract_msg.Message(io.BytesIO(item.raw_bytes or b""))
        headers = {
            "from": message.sender,
            "to": message.to,
            "cc": message.cc,
            "subject": message.subject,
            "date": str(message.date) if message.date else None,
        }
        headers = {k: v for k, v in headers.items() if v}
        item.metadata.update(headers)
        item.metadata["title"] = headers.get("subject") or "(no subject)"

        attachments = []
        for attachment in message.attachments:
            payload = attachment.data
            if not isinstance(payload, bytes):
                continue
            filename = attachment.longFilename or attachment.shortFilename or "attachment"
            attachments.append({"filename": filename, "bytes": len(payload)})
            if item.depth < config.MAX_RECURSION_DEPTH and len(payload) <= MAX_ATTACHMENT_BYTES:
                child = ExtractionItem(
                    url=f"{item.url}!/{filename}",
                    depth=item.depth + 1,
                    parent_url=item.url,
                    raw_bytes=payload,
                    metadata={"attachment_of": item.url, "filename": filename},
                )
                child.compute_content_hash()
                item.children.append(child)

        body = message.body or ""
        item.structured = {"email": {"headers": headers, "attachments": attachments}}
        item.cleaned_text = EmailHandler._render(headers, body, attachments)
        return item


def parse_message_bytes(data: bytes) -> Optional[dict]:
    """Headers only, for callers that just want to know who sent what."""
    try:
        message = email.message_from_bytes(data, policy=email.policy.default)
    except Exception:
        return None
    return {name: str(message[name]) for name in _KEEP_HEADERS if message[name] is not None}


registry.register(EmailHandler())

__all__ = ["EmailHandler", "parse_message_bytes"]
