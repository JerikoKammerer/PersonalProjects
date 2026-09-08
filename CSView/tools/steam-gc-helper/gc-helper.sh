#!/bin/sh
# Runs the helper regardless of the current directory, so that cs2mv can be
# pointed at one path with nothing to quote.
exec node "$(dirname "$0")/gc-helper.js" "$@"
