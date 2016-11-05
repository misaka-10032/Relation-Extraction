#!/usr/bin/env python

import urllib

if __name__ == '__main__':
    print 'Downloading word2vec...'
    urllib.urlretrieve("https://www.dropbox.com/s/qexbcmo8js8aapd/vec.txt?dl=0",
                       "vec.txt")
    print 'Downloading train.txt'
    urllib.urlretrieve("https://drive.google.com/uc?export=download&id=0B64C6K1Zd6EgOFdkLWs0T0hBRUU",
                       "train.txt")
