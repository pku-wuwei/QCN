# coding=utf-8
# 数据集统计
import sys
import random 
from nltk.tokenize import word_tokenize
from xml.etree.ElementTree import parse


def get_data(flist):
    nq = na = ls = lb = la = 0
    for fname in flist:
        doc = parse('c2015/' + fname + '.xml')
        for q in doc.iterfind('Question'):
            if q.attrib['QTYPE'] == 'GENERAL':
                nq += 1
                qls = word_tokenize(q.findtext('QSubject'))
                qlb = word_tokenize(q.findtext('QBody'))
                ls += len(qls)
                lb += len(qlb)
                for c in q.iterfind('Comment'):
                    al = word_tokenize(c.findtext('CBody'))
                    la += len(al)
                    na += 1
                if 10<len(qls)<15 and 10<len(qlb)<15 and 10<len(al)<15 and c.attrib['CGOLD']=='Good':
                    print(qls)
                    print(qlb)
                    print(al)
    print('# ques.{}\t# ans.{}\tavg subj.{}\tavg body.{}\tavg ans.{}'.format(nq, na, ls/nq, lb/nq, la/na))

if __name__ == '__main__':

    get_data(['train'])
    get_data(['devel'])
    get_data(['test'])

