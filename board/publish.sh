#!/bin/bash
# Upload a static board build (python -m board static site) to any rclone destination.
#
#   DEST=<remote>:<bucket or path> PUBLIC_BASE=https://<where DEST is served>/ bash board/publish.sh OUT [build_id]
#
# OUT is the static output root (media/ and one folder per build; build_id defaults to OUT/LATEST). DEST is an
# rclone remote you have configured (an S3-compatible bucket, for example) or a local directory for a rehearsal.
# The page loads its data and videos with byte-range GETs, so DEST must be served over HTTPS with CORS allowing GET
# and HEAD with a Range header from the page's origin.
#
# Layout at DEST (the same as OUT):
#   media/...           videos, goal frames and hand keypoint downloads,     Cache-Control immutable, 1 year
#                       content-addressed names
#   <build_id>/...      the build's data and its relative index.html         immutable (a new build is a new prefix);
#                       BUILD.json is not uploaded (it names local paths)
#   index.html          the build's index.public.html                         60 s in browsers, 5 min at the edge
# Order: media, then the build, then the root index.html last, so the live page never points at an object that
# is not uploaded yet. Nothing is deleted, so rolling back is re-uploading an older build's index.public.html as
# index.html (ROLLBACK=<build_id>).
#
# Optional: DRY_RUN=1 (rclone --dry-run), ALLOW_MISSING=1 (upload although some media files are not made yet),
# MEDIA_ONLY=1 (upload only media/, which can go up before the build that uses it), RCLONE (the rclone binary).
set -euo pipefail
OUT=${1:?usage: DEST=... PUBLIC_BASE=... bash board/publish.sh OUT [build_id]}
: "${DEST:?set DEST to an rclone destination, e.g. s3:my-bucket or a local directory}"
RCLONE=${RCLONE:-$(command -v rclone || true)}
[ -n "$RCLONE" ] || { echo "rclone not found (https://rclone.org/install/)"; exit 1; }
BID=${2:-$(cat "$OUT/LATEST")}
B=$OUT/$BID
RC=("$RCLONE" --transfers 32 --checkers 64 --stats 30s --stats-one-line)
DRY=""
if [ "${DRY_RUN:-0}" = 1 ]; then RC+=(--dry-run); DRY=" (dry run)"; fi
IMM=(--header-upload "Cache-Control: public, max-age=31536000, immutable")
PAGE=(--header-upload "Cache-Control: public, max-age=60, s-maxage=300"
      --header-upload "Content-Type: text/html; charset=utf-8")

media() {
  echo "== media -> $DEST/media (skips names already there)"
  "${RC[@]}" copy "$OUT/media" "$DEST/media" --ignore-existing --exclude "*.part*" --exclude "progress.json" \
    --exclude "media_log.jsonl" "${IMM[@]}"
}

if [ "${MEDIA_ONLY:-0}" = 1 ]; then
  media
  exit 0
fi

if [ -n "${ROLLBACK:-}" ]; then
  [ -f "$OUT/$ROLLBACK/index.public.html" ] || { echo "no $OUT/$ROLLBACK/index.public.html"; exit 1; }
  "${RC[@]}" copyto "$OUT/$ROLLBACK/index.public.html" "$DEST/index.html" "${PAGE[@]}"
  echo "root index.html now serves build $ROLLBACK$DRY"
  exit 0
fi

# preflight: the build exists, its public page points at this PUBLIC_BASE, and its media is all made
[ -d "$B" ] || { echo "no build $B"; exit 1; }
if [ ! -f "$B/index.public.html" ]; then
  echo "$B has no index.public.html: rebuild with python -m board static site --public-base \$PUBLIC_BASE"
  exit 1
fi
python3 - "$B/BUILD.json" "${PUBLIC_BASE:-}" "${ALLOW_MISSING:-0}" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); want = sys.argv[2].rstrip("/") + "/" if sys.argv[2] else None
have = (b.get("public_base") or "").rstrip("/") + "/"
if want and have != want:
    sys.exit(f"build {b['build_id']} was made for public base {have}, not {want}")
m = b["media"]; missing = len(b["missing_media"])
print(f"build {b['build_id']}: {b['episodes']} episodes, videos {m['videos_present']}/{m['videos']} "
      f"({m['videos_no_source']} have no source clip), frames {m['frames_present']}/{m['frames']}")
if missing and sys.argv[3] != "1":
    sys.exit(f"{missing} media files are not made yet (run python -m board static media), or set ALLOW_MISSING=1")
PY

media
echo "== build $BID -> $DEST/$BID"
"${RC[@]}" copy "$B" "$DEST/$BID" --ignore-existing --exclude "index*.html" --exclude "BUILD.json" "${IMM[@]}"
"${RC[@]}" copyto "$B/index.html" "$DEST/$BID/index.html" "${PAGE[@]}"
echo "== root index.html -> build $BID"
"${RC[@]}" copyto "$B/index.public.html" "$DEST/index.html" "${PAGE[@]}"
echo "uploaded $BID$DRY"
