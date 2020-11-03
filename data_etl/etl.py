import os
import random
import jieba
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer

"""
读取源数据
"""


def get_txt_data(txt_file):
    most_words = []
    try:
        file = open(txt_file, 'r', encoding='utf-8')
        for line in file.readlines():
            current_line = line.strip().split("\t")
            most_words.append(current_line)
        file.close()
    except:
        try:
            file = open(txt_file, 'r', encoding='gb2312')
            for line in file.readlines():
                current_line = line.strip().split("\t")
                most_words.append(current_line)
            file.close()
        except:
            try:
                file = open(txt_file, 'r', encoding='gbk')
                for line in file.readlines():
                    current_line = line.strip().split("\t")
                    most_words.append(current_line)
                file.close()
            except:
                ''
    return most_words


"""
获取停用词
"""

def load_and_merge_stopwords():
    stopwords_set = set()
    # load stopwords into set collection so as to distinct
    for root, dirs, filename_list in os.walk(r'../data/stop_words'):
        for filename in filename_list:
            file = open(root + '/' + filename, 'r')
            for line in file:
                if len(line.strip()):
                    stopwords_set.add(line.strip())
            file.close()

    print("加载停用词已完成，停用词共 %d 个" % len(stopwords_set))
    return list(stopwords_set)


def context_cut(stopwords, sentence):
    cut_words_str = ""
    cut_words_list = []
    cut_words = list(jieba.cut(sentence))
    for word in cut_words:
        if word in stopwords:
            continue
        else:
            cut_words_list.append(word)
        cut_words_str = ','.join(cut_words_list)
    return cut_words_str, cut_words_list

def do_data_etl(using_test_dataset=True):
    # 1.提取停用词
    stopwords = load_and_merge_stopwords()

    # 2.读取数据集，进行分词且过滤掉停用词
    words = []
    word_list = []
    neg_doc = []
    if using_test_dataset:
        neg_doc = get_txt_data('../data/neg_head.txt')
    else:
        neg_doc = get_txt_data('../data/neg.txt')
    for line_info in neg_doc:
        cut_words_str, cut_words_list = context_cut(stopwords, line_info[0])
        word_list.append((cut_words_str, -1))
        words.append(cut_words_list)

    neg_lena = len(word_list)
    print("加载消极情绪数据集，共 %d 条" % neg_lena)

    pos_doc = []
    if using_test_dataset:
        pos_doc = get_txt_data('../data/pos_head.txt')
    else:
        pos_doc = get_txt_data('../data/pos.txt')
    for line_info in pos_doc:
        cut_words_str, cut_words_list = context_cut(stopwords, line_info[0])
        word_list.append((cut_words_str, 1))
        words.append(cut_words_list)

    print("加载积极情绪数据集，共 %d 条" % (len(word_list) - neg_lena))

    random.shuffle(word_list)

    print("加载数据集完成，数据共 %d 条" % (len(word_list)))

    x, y = zip(*word_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size = 0.25)

    print("数据分割已完成，训练数据集共 %d 条，测试数据集共 %d 条" % (len(x_train), len(x_test)))

    # 3.提取特征向量
    # vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), max_features=500)
    vectorizer = CountVectorizer()
    return x_train, x_test, y_train, y_test, vectorizer