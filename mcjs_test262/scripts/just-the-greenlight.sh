#!/bin/bash -e

here="$(dirname "$0")"

jq -r '.[] | [(.error == null), .file_path] | @csv' <out.json

