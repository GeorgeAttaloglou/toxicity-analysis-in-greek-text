import fasttext.util
fasttext.util.download_model('el', if_exists='ignore')
ft = fasttext.load_model('./Data/cc.el.300.bin')
