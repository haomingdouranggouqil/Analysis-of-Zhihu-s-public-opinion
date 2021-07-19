# coding=utf-8
import json
import jieba
import re
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from wordcloud import WordCloud
import pandas as pd
#from jieba.analyse import *
from gensim import corpora, models
import jieba.posseg as jp, jieba
def seg_word(path):
    #因需求输入不同，故从头开始提取数据，放入列表，由pku分词
    cleantxt = []
    context = []
    df = pd.read_csv(path)
    for i in df['content']:
        if i != 'content':
            context.append(i)
    for poem in context:
        sentence_list = re.split(r'[，。！？、（）【】<>《》—“”…\n]', poem)
        for s in sentence_list:
            cleantxt.append(s)
    #初始化pku分词
    word_list = []
    for sentence in cleantxt:
        word = jieba.cut(sentence)
        word_list += word
    return word_list

def character_sort(character_list):
    #传入文字列表，输出对应排序字典
    character_dict = {}
    for character in character_list:
        if character not in character_dict:
            character_dict[character] = 1
        else:
            character_dict[character] += 1

    sorted_character = sorted(character_dict.items(), key=lambda x:x[1],  reverse=True)
    return sorted_character

def sort_word(path):
    #传入路径，输出词频
    word_list = seg_word(path)
    clean_word_list = []
    for word in word_list:
        if len(word) == 1:
            pass
        else:
            clean_word_list.append(word)
    sorted_list = character_sort(clean_word_list)
    return sorted_list

def show_outcome(outcome_list, n):
    #传入字频结果列表，输出前n个结果
    for i in range(n):
        sort_show = outcome_list[i][0] + "    " + str(outcome_list[i][1])
        print(sort_show)

def word_main(path):
    sj_word_list = sort_word(path)
    print('前100非单字词频')
    show_outcome(sj_word_list, 100)
    sj_pku_seg = seg_word(path)
    word_cloud(sj_pku_seg)

def word_cloud(pku_word):
    #传入pku分词结果，生成字云
    sj_word = ' '.join(pku_word)
    pku_cloud = WordCloud(scale = 32, font_path = "青鸟华光简隶变.ttf").generate(sj_word)
    plt.imshow(pku_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.png', dpi = 4000)
    plt.show()


def cal_emo(test):
    poemstr = ''
    poem_sentence_list = []
    for i in test:
        poemstr += i
    poem_sentence_list = re.split(r'[，。！？]', poemstr)
    del(poem_sentence_list[-1])
    emotion = 0
    for sentence in poem_sentence_list:
        s = SnowNLP(sentence)
        emotion += s.sentiments
    emotion = emotion / len(poem_sentence_list)
    return emotion

def mean_emo(poem):
    all_emo = 0
    for i in poem:
        all_emo += cal_emo(i)
    all_emo = all_emo / len(poem)
    return all_emo

def topic(poem):
    word_list = seg_word(poem)
    poemstr = ' '.join(word_list)
    keyword = []
    for i in extract_tags(poemstr, topK = 30, withWeight=True):
        print(i[0] + '    ' + str(i[1]))




#前100词频
#word_main('高校九月份开学还会实施全封闭管理吗.csv')
#topic('如何看待天津大学仍然不让学生出校门.csv')

c = []
df = pd.read_csv('高校九月份开学还会实施全封闭管理吗.csv')
for i in df['content']:
    if i != 'content':
        c.append(i)


#引入停用词
path = 'cn_stopwords.txt'
f = open(path, 'r', encoding='UTF-8',errors = 'ignore')
stop = f.read()
stopword = re.split(r'[\n]', stop)


word_list = []
for poem in c:
    cleantxt = []
    sentence_list = re.split(r'[，。！？、（）【】<>《》—“”…\n]', poem)
    for s in sentence_list:
        word = jieba.cut(s)
        for w in word:
            if w in stopword:
                pass
            #elif w == ' ' or w == '天大' or w == '学校' or w == '学生' or len(w) == 1:
            elif w == ' ' or w == '学校' or w == '学生' or len(w) == 1:
                pass
            else:
                cleantxt.append(w)
    word_list.append(cleantxt)



dictionary = corpora.Dictionary(word_list)
# 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
corpus = [dictionary.doc2bow(words) for words in word_list]
# lda模型，num_topics设置主题的个数
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
# 打印所有主题，每个主题显示5个词
for topic in lda.print_topics(num_words=100):
    print(topic)
    print('----------' * 10)

'''
for n in range(16):
    n *= 150
    cleantxt = []
    for poem in c[n : n + 150]:
        sentence_list = re.split(r'[，。！？、（）【】<>《》—“”…\n]', poem)
        for s in sentence_list:
            cleantxt.append(s)
    #初始化pku分词
    word_list = []
    for sentence in cleantxt:
        word = jieba.cut(sentence)
        word_list += word
    sj_word = ' '.join(word_list)
    pku_cloud = WordCloud(scale = 32, font_path = "青鸟华光简隶变.ttf").generate(sj_word)
    plt.imshow(pku_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(str(n)+'.png', dpi = 4000)
    print(n)


'''

'''
c = [['name', 'gender','follower_count','voteup','cmt_count','content', 'character_num', 'emotion1', 'relief', 're1', 'sight']]
df = pd.read_csv('如何看待天津大学仍然不让学生出校门.csv')
s = 2422
for row in df.itertuples():
    r = []
    a = getattr(row, 'gender')
    if a != 'gender':
        r.append(getattr(row, 'name'))
        r.append(getattr(row, 'gender'))
        r.append(getattr(row, 'follower_count'))
        r.append(getattr(row, 'voteup'))
        r.append(getattr(row, 'cmt_count'))
        r.append(getattr(row, 'content'))
        r.append(getattr(row, 'character_num'))
        r.append(getattr(row, 'emotion1'))
        r.append(getattr(row, 'relief'))
        r.append(getattr(row, 're1'))
        r.append(s)
        s -= 1
        c.append(r)

dataframe = pd.DataFrame(c)
dataframe.to_csv('t.csv', mode='a', index=False, sep=',', header=False)
'''
