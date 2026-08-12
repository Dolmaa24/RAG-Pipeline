import logging

from charset_normalizer import from_bytes

from models import ExtractionItem, Stage

logger = logging.getLogger("TextDecoder")


class TextDecoder:
    @staticmethod
    def decode(item: ExtractionItem) -> ExtractionItem:
        """Turn raw bytes into text using statistical + BOM charset detection.

        Only meaningful for the web path. The media path produces text directly
        (a Whisper transcript) and must not be sent through here.
        """
        if not item.raw_bytes:
            return item.fail(Stage.DECODE, "No raw bytes to decode")

        try:
            best_guess = from_bytes(item.raw_bytes).best()
        except Exception as exc:  # charset-normalizer can throw on pathological input
            logger.warning(f"Charset detection failed, falling back to UTF-8: {exc}")
            best_guess = None

        if best_guess:
            item.decoded_text = str(best_guess)
        else:
            # Fallback: UTF-8, replacing corrupt sequences
            item.decoded_text = item.raw_bytes.decode("utf-8", errors="replace")

        if not item.decoded_text.strip():
            return item.fail(Stage.DECODE, "Decoded document is empty")

        return item
