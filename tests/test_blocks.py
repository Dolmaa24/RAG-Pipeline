"""Recognising anti-bot walls -- and, just as much, not mistaking pages for them.

The module is conservative by construction: a false positive stops a legitimate
job, while a false negative only produces an ordinary retry. Both directions are
tested here, because the tempting fix for a missed wall is a looser marker, and
a looser marker is what starts refusing real pages.
"""

from __future__ import annotations

from pipeline.fetch.blocks import block_guidance, detect_block

ORDINARY = b"<html><body><h1>Quarterly report</h1><p>Revenue rose.</p></body></html>"


def test_an_ordinary_page_is_not_a_block():
    assert detect_block(status=200, headers={}, body=ORDINARY) is None


def test_a_plain_server_error_is_not_a_block():
    """A 503 is retried; a wall is not. Confusing them breaks both behaviours."""
    assert detect_block(status=503, headers={}, body=b"<h1>Service Unavailable</h1>") is None


def test_a_long_document_mentioning_a_marker_is_not_a_block():
    """A document about access denial is a document, not a wall."""
    body = b"you have been blocked" + b" filler." * 40_000
    assert detect_block(status=200, headers={}, body=body) is None


def test_a_vendor_header_is_believed_at_200():
    signal = detect_block(status=200, headers={"cf-mitigated": "challenge"}, body=ORDINARY)
    assert signal is not None and signal.source == "header"


def test_datadome_is_recognised_by_its_header_alone():
    signal = detect_block(status=200, headers={"X-DataDome": "protected"}, body=ORDINARY)
    assert signal is not None and signal.name == "datadome"


def test_the_cloudflare_interstitial_is_caught_at_200():
    """The case the module exists for: a challenge that answers 200."""
    signal = detect_block(status=200, headers={}, body=b"<title>Just a moment...</title>")
    assert signal is not None
    assert signal.name == "cloudflare interstitial"


def test_a_weak_marker_needs_a_refusal_status():
    body = b"<h1>Are you a robot</h1>"
    assert detect_block(status=200, headers={}, body=body) is None
    assert detect_block(status=403, headers={}, body=body) is not None


#: Trimmed from the live response of an .ac.in host that began challenging this
#: pipeline mid-session. HTTP 200, no vendor header, and served just as readily
#: for /robots.txt as for a page.
APPLIANCE_CAPTCHA = (
    b"<html><body><title>Validation request</title>"
    b'<h3 align="center">User validation required to continue..</h3><hr>'
    b"Please type the text you see in the image into the text box and submit"
    b'<p><img src = "/captcha.gif"></p>'
    b'<form name="input" action="/captcha_resp" method="POST">'
    b'<input type="text" name="captcha_resp_txt" /></form></body></html>'
)


def test_an_appliance_captcha_is_caught_at_status_200():
    """No refusal status and no vendor header -- the body is the only evidence."""
    signal = detect_block(status=200, headers={}, body=APPLIANCE_CAPTCHA)
    assert signal is not None
    assert signal.name == "captcha interstitial"


def test_the_same_wall_is_caught_when_served_as_robots_txt():
    """The dangerous shape, and the reason this signature had to be standalone.

    Unrecognised, a challenge page parses as a robots file with no rules, and
    the crawler concludes the site permits everything -- a permission derived
    from a page that was refusing it.
    """
    assert detect_block(status=200, headers={}, body=APPLIANCE_CAPTCHA) is not None


def test_either_marker_alone_is_enough():
    for fragment in (b"User validation required to continue", b'name="captcha_resp_txt"'):
        assert detect_block(status=200, headers={}, body=b"<html>" + fragment + b"</html>")


def test_a_page_that_merely_discusses_captchas_is_not_a_wall():
    """Why bare "captcha" is not a marker, and must not be added as one."""
    article = (
        b"<html><body><h1>How CAPTCHA systems work</h1><p>A captcha asks the "
        b"user to prove they are human. Modern captcha designs avoid distorted "
        b"text, and some sites drop the captcha entirely.</p></body></html>"
    )
    assert detect_block(status=200, headers={}, body=article) is None


def test_the_guidance_offers_routes_around_and_refuses_evasion():
    text = block_guidance("example.com")
    assert "example.com" in text
    assert "Official API" in text
    assert "Do not respond by rotating addresses" in text
    assert "solving" in text
