import csv, pprint


class convert_conll2spacy(object):

    def __init__(self, file):
        self.file = file

    def convert(self):
        print("Start Conversion")
        with open(self.file, 'r') as devset:
            content = csv.reader(devset, delimiter=' ', skipinitialspace=True, quotechar=None)

            text_as_list = []
            sentence_as_list = []
            entities = []
            sentences_as_plain_text = ""
            i = 0
            tokenized_list = []

            for row in content:
                if len(row) == 0:
                    tokenized_list.append(" ")
                else:
                    tokenized_list.append(row[0])
                if len(row) == 5:
                    if 'B-MISC' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'B-MISC'))
                    if 'I-MISC' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'I-MISC'))
                    if 'B-LOC' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'B-LOC'))
                    if 'I-LOC' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'I-LOC'))
                    if 'B-ORG' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'B-ORG'))
                    if 'I-ORG' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'I-ORG'))
                    if 'B-PER' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'B-PER'))
                    if 'I-PER' in row[4]:
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'I-PER'))
                    if row[4] == 'O':
                        start = i
                        end = i+len(row[0])
                        entities.append((start, end, 'O'))
                    sentence_as_list.append(row[0])
                    i += len(row[0])+1

                elif len(row) == 0:
                    i = 0
                    sentence = " ".join(sentence_as_list)
                    sentences_as_plain_text += sentence
                    add_sent_ne_to_list = (sentence, entities)
                    text_as_list.append(add_sent_ne_to_list)
                    sentence_as_list = []
                    entities = []

        #pprint.pprint(text_as_list)
        print("Conversion done!")
        return text_as_list, sentences_as_plain_text, tokenized_list

Cv = convert_conll2spacy("/home/tom/Dokumente/UniSoSe2017/softwarepra/ner/corpora/conll2003/deuutf.testa")
Cv.convert()