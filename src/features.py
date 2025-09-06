# src/features.py
from __future__ import annotations
import re
import math
from urllib.parse import urlparse
import tldextract

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "ebayisapi", "wp-admin",
    "banking", "confirm", "account", "support", "billing", "password",
]

IP_REGEX = re.compile(r"^(\d{1,3}\.){3}\d{1,3}(:\d+)?$")

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def count_chars(s: str, charset: str) -> int:
    cs = set(charset)
    return sum(ch in cs for ch in s)

SPECIAL = "-_.@/:?=&%#"
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def has_ip_host(netloc: str) -> bool:
    # strip credentials if present
    host = netloc.split("@")[ -1 ]
    host = host.split(":")[0]
    return bool(IP_REGEX.match(host))

def features_from_url(url: str) -> dict:
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    ext = tldextract.extract(url)
    netloc = parsed.netloc or f"{ext.subdomain}.{ext.registered_domain}".strip('.')
    path = parsed.path or ""
    query = parsed.query or ""
    scheme = parsed.scheme or ""

    u = url or ""
    url_len = len(u)

    num_digits = count_chars(u, DIGITS)
    num_letters = count_chars(u, LETTERS)
    num_special = count_chars(u, SPECIAL)

    num_dots = u.count('.')
    num_hyphens = u.count('-')
    num_slashes = u.count('/')
    num_q = u.count('?')
    num_eq = u.count('=')
    num_at = u.count('@')

    subdomain_count = (ext.subdomain.count('.') + 1) if ext.subdomain else 0
    tld_len = len(ext.suffix or "")

    entropy = shannon_entropy(u)

    susp_count = sum(1 for w in SUSPICIOUS_WORDS if w in u.lower())

    feats = {
        "url_length": url_len,
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_special": num_special,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_slashes": num_slashes,
        "num_question": num_q,
        "num_equal": num_eq,
        "num_at": num_at,
        "uses_https": int(scheme.lower() == "https"),
        "has_ip": int(has_ip_host(netloc)),
        "subdomain_count": subdomain_count,
        "tld_length": tld_len,
        "path_length": len(path),
        "query_length": len(query),
        "entropy": float(entropy),
        "pct_digits": num_digits / (url_len + 1e-9),
        "pct_special": num_special / (url_len + 1e-9),
        "suspicious_words": susp_count,
    }
    return feats

def batch_extract(urls: list[str]) -> list[dict]:
    return [features_from_url(u) for u in urls]
