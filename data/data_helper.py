# coding=utf-8
# 数据集统计
import sys
import random 
from nltk.tokenize import word_tokenize
from xml.etree.ElementTree import parse


def get_data(flist):
    nq = na = ls = lb = la = 0
    for fname in flist:
        doc = parse('xml/' + fname + '.xml')
        for t in doc.iterfind('Thread'):
            q = t.find('RelQuestion')
            nq += 1
            sl = word_tokenize(q.findtext('RelQSubject'))
            bl = word_tokenize(q.findtext('RelQBody'))
            ls += len(sl)
            lb += len(bl)
            for c in t.iterfind('RelComment'):
                al = word_tokenize(c.findtext('RelCText'))
                la += len(al)
                na += 1
            if all([10<len(i)<15 for i in [sl, bl, al]]) and c.attrib['RELC_RELEVANCE2RELQ']=='Good':
                print(sl)
                print(bl)
                print(al)

    print('# ques.{}\t# ans.{}\tavg subj.{}\tavg body.{}\tavg ans.{}'.format(nq, na, ls/nq, lb/nq, la/na))

if __name__ == '__main__':

    get_data(['15train', '15dev', '15test', '16train1', '16train2'])
    get_data(['16dev'])
    get_data(['16test'])

