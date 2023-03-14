import jieba

doc_word_freq = {}
word_id_map = {}
f = open('data/corpus/' + '微博' + '_vocab.txt', 'w',encoding='utf-8')
vocab = open('data/corpus/' + '微博' + '_vocab.txt', 'r',encoding='utf-8').readlines()
vocab_size = len(vocab)
for i in range(vocab_size):
    word_id_map[vocab[i]] = i
f = open('data/corpus/' + '微博' + '_shuffle.txt', 'r',encoding='utf-8')
shuffle_doc_words_list = f.readlines()
for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = jieba.lcut(doc_words)
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1