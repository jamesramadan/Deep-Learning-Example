#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jamesramadan
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu' #uses GPU if available, otherwise defaults to CPU for processing

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me and come here".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("lo olvide".split(), "SPANISH"),
        ("Yo creo que si".split(), "SPANISH"),
        ("he ate too much cake on his birthday".split(), "ENGLISH"),
        ("she did not spend her money on a card".split(), "ENGLISH"),
        ("no it is not a good idea to get lost at sea".split(), "ENGLISH"),
        ("it is lost on me".split(), "ENGLISH"),
        ("oh yes man smile at my funny jokes".split(), "ENGLISH"),
        ("a mi tambien".split(), "SPANISH"),
        ("no le gusta".split(), "SPANISH"),
        ("no problem brother and sister".split(), "ENGLISH"),
        ("puedo comprar esto".split(), "SPANISH"),
        ("if you insist on paint the house black".split(), "ENGLISH"),
        ("dance to the rhythm of the night".split(), "ENGLISH"),
        ("one two three four five six seven eight nine ten hundred million".split(), "ENGLISH"),
        ("uno dos tres cuatro cinco seis sieta ocho nueve dies cien mil".split(), "SPANISH"),
        ("por supuesto o por la tarde".split(), "SPANISH"),
        ("hay mas libros cual puedemos leer".split(), "SPANISH"),
        ("en dies minutos es la fin de la semana".split(), "SPANISH"),
        ("no bebes agua ni leche".split(), "SPANISH"),
        ("he passed three times".split(), "ENGLISH"),
        ("spider cow monkey donkey seal giraffe dog cat elephant".split(), "ENGLISH"),
        ("house knife fork spoon".split(), "ENGLISH"),
        ("it's going to rain so bring your umbrella".split(), "ENGLISH"),
        ("in out new old over under up down left right fast slow tight loose hard soft round skinny hot cold fat pencil man dude".split(), "ENGLISH"),
        ("my husband wife daughter brother son sister aunt and uncle will all be there".split(), "ENGLISH"),
        ("we didn't start the fire it was always burning since the world has been turning".split(), "ENGLISH"),
        ("i have to break up with you this relationship is finished".split(), "ENGLISH"),
        ("can't touch this".split(), "ENGLISH"),
        ("cant stop wont stop".split(), "ENGLISH"),
        ("escucha bien tambien tenga cuidado".split(), "SPANISH"),
        ("tengo mas or menos de dies dolares".split(), "SPANISH"),
        ("a ti no tocas tampoco el perro".split(), "SPANISH"),
        ("a vez en cuando".split(), "SPANISH"),
        ("es dificil y facil no puedo explicar".split(), "SPANISH"),
        ("fiesta prueba amigo amiga bailamos".split(), "SPANISH"),
        ("dude wheres my car".split(), "ENGLISH"),
        ("you will have to see it to believe it trust me".split(), "ENGLISH"),
        ("we went to school together but the teacher said we needed to study more and play less".split(), "ENGLISH"),
        ("queue that song".split(), "ENGLISH"),
        ("work hard party harder".split(), "ENGLISH"),
        ("dale ahora".split(), "SPANISH"),
        ("i want a new car".split(), "ENGLISH"),
        ("no not today and especially not on my birthday".split(), "ENGLISH"),
        ("you also wanna play games but I have a job".split(), "ENGLISH"),
        ("take my breath away".split(), "ENGLISH"),
        ("quiero una nueva espousa porque mi espousa ahora no es intellegente".split(), "SPANISH"),
        ("que dice en la discoteca".split(), "SPANISH"),
        ("mi gente".split(), "SPANISH"),
        ("bonjour bienvenue oui non merci".split(), "FRENCH"),
        ("bonne après-midi".split(), "FRENCH"),
        ("bonne apres-midi".split(), "FRENCH"),
        ("je m'appelle mondly".split(), "FRENCH"),
        ("je suis ravi de vous rencontrer".split(), "FRENCH"),
        ("comment ça va".split(), "FRENCH"),
        ("comment ca va".split(), "FRENCH"),
        ("ou es-tu".split(), "FRENCH"),
        ("j’aimerais une bière".split(), "FRENCH"),
        ("j’aimerais une biere".split(), "FRENCH"),
        ("je suis désolé".split(), "FRENCH"),
        ("je suis desole".split(), "FRENCH"),
        ("au revoir".split(), "FRENCH"),
        ("je suis heureuse".split(), "FRENCH"),
        ("amusez-vous bien".split(), "FRENCH"),
        ("ou se trouve le bureau de poste".split(), "FRENCH"),
        ("avez-vous la taille en dessous".split(), "FRENCH"),
        ("je voudrais payer avec ma carte de credit.".split(), "FRENCH"),
        ("je t'aime mon amour".split(), "FRENCH"),
        ("parlez-vous anglais".split(), "FRENCH"),
        ("je ne parle pas francais".split(), "FRENCH"),
        ]


test_text = input("Enter a phrase in Spanish, English, or French: ")
test_data = [(test_text.split(), "")]
print('processing...')

word_to_ix = {}
for i, phrases in enumerate(data + test_data):
    for word in phrases[0]:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 3

class BoWClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)


label_to_ix = {"SPANISH": 0, "ENGLISH": 1, "FRENCH": 2}

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        probs = np.exp(log_probs)


loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(120):
    for instance, label in data:

        model.zero_grad()

        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)
        log_probs = model(bow_vec)

        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        probs = np.exp(log_probs).numpy()
        span_prob = probs[0][0]
        eng_prob = probs[0][1]
        french_prob = probs[0][2]
        language = 'Spanish'
        prob_stored = span_prob
        if (eng_prob >= span_prob and eng_prob >= french_prob):
            language = 'English'
            prob_stored = eng_prob
        if (french_prob >= span_prob and french_prob >= eng_prob):
            language = 'French'
            prob_stored = french_prob
        print('I am ', round((prob_stored * 100),2), '% ', 'sure that you typed ', language, '.')


