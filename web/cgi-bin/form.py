#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


import cgi
import html


import numpy as np
import math
import Levenshtein 
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import jaccard_score

def shj(sent1, sent2):
    
    vect = CountVectorizer().fit_transform([sent1, sent2])
    cos_sim = cosine_similarity(vect.toarray())
    eucl_dist = euclidean_distances(vect.toarray())
    jac_sc = jaccard_score(vect.toarray()[0], vect.toarray()[1], average='micro')
    lev_dist =  Levenshtein.distance(sent1, sent2)/(len(sent1)+len(sent2))

    x = np.array([
        cos_sim[1][0], 
        eucl_dist[1][0], 
        jac_sc, 
        lev_dist
        ])
    w = np.array([ 2.9195331 , -0.1007705 ,  2.16236057, -2.47438075])
    y = np.dot(x, w)
    target = round(1/(1+math.exp(-y)))
    answer = 'Фраза схожа' if target == 1 else 'Фраза не схожа'
    return answer




form = cgi.FieldStorage()
text1 = form.getfirst("TEXT_1", "не задано")
text2 = form.getfirst("TEXT_2", "не задано")
text1 = html.escape(text1)
text2 = html.escape(text2)

print("Content-type: text/html\n")
print("""<!DOCTYPE HTML>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Проверка схожести фраз</title>
        </head>
        <body>""")


print("<p>Результат: {}</p>".format(shj(text1, text2)))

print("""</body>
        </html>""")



