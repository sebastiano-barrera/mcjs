#!/bin/sh

exec npx tailwindcss -i data/input/style.css -o data/assets/style.css --watch
