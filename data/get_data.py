
import urllib

if __name__ == '__main__':
    download_list = [("https://www.dropbox.com/s/qexbcmo8js8aapd/vec.txt?dl=1", "vec.txt"),
                     ("https://drive.google.com/uc?export=download&id=0B64C6K1Zd6EgOFdkLWs0T0hBRUU",
                                        "train.txt"),
                     ("https://www.dropbox.com/s/clmzhqapg92ue75/id2vec.bin?dl=1", "id2vec.bin"),
                     ("https://www.dropbox.com/s/hkh5rmgsfquhj3q/instances.bin?dl=1", "instances.bin"),
                     ("https://www.dropbox.com/s/tt3cvqg67inihnr/word2id.bin?dl=1", "word2id.bin")
                     ]
    for url, filename in download_list:
        print 'Downloading %s' % filename
        urllib.urlretrieve(url, filename)
