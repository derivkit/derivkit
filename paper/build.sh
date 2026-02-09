#!/bin/sh

#
# Typesets a draft paper using Open Journals' inara build recipe
# (see https://github.com/openjournals/inara). Builds the container
# if it isn't already present in the current working directory.
# Run this script from the `paper` directory.
#

[ ! -f inara.sif ] && apptainer build inara.sif docker://openjournals/inara

APPTAINERENV_JOURNAL=joss
apptainer run inara.sif paper.md
