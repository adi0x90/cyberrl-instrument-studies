#!/usr/bin/env bash
# fetch.sh <slug> <url> — fetch one source, save raw HTML + extracted text, print status/bytes.
# No snippets: this fetches the actual document body.
set -uo pipefail
SLUG="$1"; URL="$2"
DIR="/Users/adi/research/cyber-rl-experiments/2026-11-27-envfactory/evidence/sources"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
TMP_HTML="$DIR/$SLUG.raw.html"
TXT="$DIR/$SLUG.txt"

CODE=$(curl -sSL --max-time 60 -A "$UA" -H "Accept-Language: en-US,en;q=0.9" \
  -o "$TMP_HTML" -w "%{http_code}" "$URL" 2>"$DIR/$SLUG.curlerr")
CURL_RC=$?
CTYPE=$(file -b --mime-type "$TMP_HTML" 2>/dev/null || echo unknown)
RAWBYTES=$(stat -f%z "$TMP_HTML" 2>/dev/null || echo 0)

# Extract text: if PDF, pdftotext; else strip HTML via python stdlib.
if [ "$CTYPE" = "application/pdf" ]; then
  pdftotext -layout "$TMP_HTML" "$TXT" 2>/dev/null
else
  python3 - "$TMP_HTML" "$TXT" <<'PY'
import sys, re, html
src, dst = sys.argv[1], sys.argv[2]
raw = open(src, 'rb').read().decode('utf-8', 'replace')
# capture title + meta dates before stripping
head = raw[:200000]
title = re.search(r'<title[^>]*>(.*?)</title>', head, re.S|re.I)
metas = re.findall(r'<meta[^>]+>', head, re.I)
keep = []
for m in metas:
    if re.search(r'(published_time|modified_time|datePublished|dateModified|citation_|dc\.date|description|og:title|twitter:title)', m, re.I):
        keep.append(m.strip())
raw = re.sub(r'(?is)<(script|style|noscript|svg)\b.*?</\1>', ' ', raw)
raw = re.sub(r'(?is)<!--.*?-->', ' ', raw)
raw = re.sub(r'(?i)</(p|div|li|h[1-6]|tr|section|article|br)>', '\n', raw)
raw = re.sub(r'(?i)<br\s*/?>', '\n', raw)
txt = re.sub(r'(?s)<[^>]+>', ' ', raw)
txt = html.unescape(txt)
txt = re.sub(r'[ \t\xa0]+', ' ', txt)
txt = re.sub(r'\n\s*\n+', '\n\n', txt)
txt = txt.strip()
hdr = []
if title: hdr.append('TITLE: ' + html.unescape(title.group(1)).strip())
hdr.append('SOURCE-URL: ' + __import__('os').environ.get('SRC_URL',''))
hdr.append('META:')
hdr.extend('  '+k for k in keep[:12])
out = '\n'.join(hdr) + '\n\n===== BODY =====\n\n' + txt
open(dst,'w').write(out[:200*1024])
PY
fi

TXTB=$(stat -f%z "$TXT" 2>/dev/null || echo 0)
TXTBYTES=$(wc -c < "$TXT" 2>/dev/null | tr -d ' ')
echo "slug=$SLUG"
echo "http_status=$CODE curl_rc=$CURL_RC ctype=$CTYPE raw_bytes=$RAWBYTES"
echo "saved=$TXT text_bytes=$TXTBYTES"
if [ $CURL_RC -ne 0 ]; then echo "CURL_ERROR: $(cat "$DIR/$SLUG.curlerr" 2>/dev/null)"; fi
# truncation flag
if [ "$RAWBYTES" -lt 3000 ]; then echo "WARN: tiny body — possible JS wall / error page"; fi
head -c 400 "$TXT" 2>/dev/null; echo
