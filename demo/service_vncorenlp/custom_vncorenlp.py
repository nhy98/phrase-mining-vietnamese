import requests
from requests.exceptions import RequestException
from service_vncorenlp.config_vncorenlp import HOST, PORT, ANNOTATOR
import os

class VnCoreNLP():
    def __init__(self):
        self.url = f"http://{HOST}:{PORT}"
        self.timeout = 30
        self.annotators = set(ANNOTATOR.split(","))
        if not self.is_alive():
            raise ConnectionError("Run start_vncorenlp.sh to enable service VnCoreNLP")

    def is_alive(self):
        # Check if the server is alive
        try:
            response = requests.get(self.url, timeout=self.timeout)
            return response.ok
        except RequestException:
            pass
        return False

    def __get_annotators(self):
        # Get list of annotators from the server
        response = requests.get(self.url + '/annotators', timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def annotate(self, text, annotators=None):
        if isinstance(annotators, str):
            assert self.annotators.issuperset(annotators.split(
                ',')), 'Please ensure that the annotators "%s" are being used on the server.' % annotators
        data = {
            'text': text.encode('UTF-8'),
            'props': annotators
        }
        response = requests.post(self.url + '/handle', data=data, timeout=self.timeout)
        response.raise_for_status()
        response = response.json()
        while not response['status']:
            response = requests.post(self.url + '/handle', data=data, timeout=self.timeout)
            response.raise_for_status()
            response = response.json()
        assert response['status'], response['error']
        del response['status']
        return response

    def tokenize(self, text):
        sentences = self.annotate(text, annotators='wseg')['sentences']
        return [[w['form'] for w in s] for s in sentences]

    def pos_tag(self, text):
        sentences = self.annotate(text, annotators='wseg,pos')['sentences']
        return [[(w['form'], w['posTag']) for w in s] for s in sentences]

    def ner(self, text):
        sentences = self.annotate(text, annotators='wseg,pos,ner')['sentences']
        return [[(w['form'], w['nerLabel']) for w in s] for s in sentences]

    def dep_parse(self, text):
        sentences = self.annotate(text, annotators='wseg,pos,ner,parse')['sentences']
        # dep, governor, dependent
        return [[(w['depLabel'], w['head'], w['index']) for w in s] for s in sentences]

    def detect_language(self, text):
        return self.annotate(text, annotators='lang')['language']
