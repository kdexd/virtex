#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

CORENLP=stanford-corenlp-full-2015-12-09
SPICELIB=SPICE-1.0/lib

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget https://panderson.me/images/SPICE-1.0.zip
wget http://nlp.stanford.edu/software/$CORENLP.zip
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip

echo "Unzipping..."

unzip SPICE-1.0.zip

unzip $CORENLP.zip -d $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0.jar $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0-models.jar $SPICELIB/
rm -f stanford-corenlp-full-2015-12-09.zip
rm -rf $SPICELIB/$CORENLP/

rm -rf SPICE-1.0.zip
