"""API de categorização de produtos usando machine learning"""


import pickle
import os

import numpy as np

from flask import Flask
from flask import request

api = Flask(__name__)

model_path = os.environ['MODEL_PATH']
model_pkl = open(model_path, 'rb')
model = pickle.load(model_pkl)


def category_converter(integer):
    """Converte as categorias numéricas de saída do modelo em categorias textuais"""
    if integer == 0:
        category = "Decoração"
    if integer == 1:
        category = "Papel e Cia"
    if integer == 2:
        category = "Outros"
    if integer == 3:
        category = "Bebê"
    if integer == 4:
        category = "Lembrancinhas"
    if integer == 5:
        category = "Bijuterias e Jóias"
    return category

vectorized_converter = np.vectorize(category_converter)

@api.route("/v1/categorize", methods=["POST"])
def categorize():
    """API de categorização utilizando o modelo Perceptron()"""

    # Load input
    body = request.json

    # Error handling
    if not "products" in body:
        return { "error": "json field 'products' does not exist"}, 400
    products = body["products"]

    if type(products) != list:
        return { "error": "'products' must be a list of objects"}, 400

    # Extrair os nomes de produto do método POST e formatar de acordo com as regras
    # estabelecidas na pipeline de treinamento. Observe que nem todos os jsons
    # terão todas as keys, então optou por se fazer um exception handling
    inputs = []
    for product in products:
        # Error handling (produtos individuais)
        if type(product) != dict:
            return { "error": "'products' must be a list of objects"}, 400

        empty_indicator = 0

        try:
            concatenated_tags = product["concatenated_tags"]
        except KeyError:
            concatenated_tags = ""
            #indicar que as tags estão vazias
            empty_indicator+=1

        try:
            title = product["title"]
        except KeyError:
            title = ""
            #indicar que o título está vazio
            empty_indicator+=1

        try:
            query = product["query"]
        except KeyError:
            query = ""
            #indicar que query está vazia
            empty_indicator+=1

        #Error handling (se uma json query estiver vazia)
        if empty_indicator == 3:
            return {
                "error": "product json queries must contain at least one of the " +
                "following elements: 'query', 'concatenated_tags' or 'title'"
                }, 400

        # O input da heurística de machine learning tem o formato
        # query + ' ' + title + ' ' + concatenated_tags
        input_ = query + ' ' + title + ' ' + concatenated_tags
        inputs.append(input_)
        integer_categories = model.predict(inputs) # Numpy array
        # Converter as categorias para nomes. Os elementos inteiros serão mapeados
        # para categorias em string com a iteração vetorizada rápida da biblioteca numpy

        str_categories = vectorized_converter(integer_categories)
        str_list = str_categories.tolist()

    return {"categories": str_list}
