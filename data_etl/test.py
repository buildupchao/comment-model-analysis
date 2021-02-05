from data_etl import etl

sentence = '我同意另外一个网友的观点：“阿里巴巴总体来说市场价值真的很重要。美中不足的就是网销宝功能是在有点坑人哈！这也是为什么很多人做了每单的重要原因”。很多朋友问过我对阿里巴巴的评价，在此我也简单谈一下，在阿里巴巴想做好的话，先好好的潜下心来去研究它的规则，熟悉游戏规则的人永远都是赢家。连规则都不懂，就想玩转它不可能。每件事物都有两面性，阿里巴巴更是。很多人在里面都很惨，但是也有很多人做的很好，这就是很好的例子，你把规则搞懂了，弄清楚了，才能玩转它，要不，你的下场会很惨，而不是好不好的问题。'

stopwords = etl.load_and_merge_stopwords()
cut_words_str, cut_words_list = etl.context_cut(stopwords, sentence)

print(sentence)
print((cut_words_str, 1))