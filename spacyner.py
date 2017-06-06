from __future__ import unicode_literals, print_function

import json
import pathlib
import random

import spacy
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer
from spacy.tagger import Tagger
from spacy.tokens import Doc

from convert_germaeval2spacy import convert_germaeval2spacy
from convert_conll2spacy import convert_conll2spacy


def train_ner(nlp, train_data, entity_types):
    # Add new words to vocab.
    for raw_text, _ in train_data:
        doc = nlp.make_doc(raw_text)
        for word in doc:
            _ = nlp.vocab[word.orth]

    # Train NER.
    ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = nlp.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            ner.update(doc, gold)
    return ner


def save_model(ner, model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    with (model_dir / 'config.json').open('wb') as file_:
        data = json.dumps(ner.cfg)
        if isinstance(data, unicode):
            data = data.encode('utf8')
        file_.write(data)
    ner.model.dump(str(model_dir / 'model'))
    if not (model_dir / 'vocab').exists():
        (model_dir / 'vocab').mkdir()
    ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
    with (model_dir / 'vocab' / 'strings.json').open('w', encoding='utf8') as file_:
        ner.vocab.strings.dump(file_)


def main(tagged_output, traindata, testdata, traindataformat="", testdataformat="", model_dir=None):
    nlp = spacy.load('de', parser=False, entity=False, add_vectors=False)

    # v1.1.2 onwards
    if nlp.tagger is None:
        print('---- WARNING ----')
        print('Data directory not found')
        print('please run: `python -m spacy.en.download --force all` for better performance')
        print('Using feature templates for tagging')
        print('-----------------')
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    if traindataformat == 'germeval':
        Cv = convert_germaeval2spacy(traindata)
        train_data = Cv.convert()[0]
        ner = train_ner(nlp, train_data,
                        ['B-OTH', 'I-OTH', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'O'])

    elif traindataformat == 'conll':
        Cv = convert_conll2spacy(traindata)
        train_data = Cv.convert()[0]
        ner = train_ner(nlp, train_data,
                        ['B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'O'])

    if testdataformat == 'germeval':
        Cv_test = convert_germaeval2spacy(testdata)
        test_data = Cv_test.convert()[2]

    elif testdataformat == 'conll':
        Cv_test = convert_conll2spacy(testdata)
        test_data = Cv_test.convert()[2]

    #doc = nlp.make_doc(test_data)
    doc = Doc(nlp.vocab, words=test_data)
    ner(doc)
    nlp.tagger(doc)
    i = 0
    testfilespacygermeval = open(tagged_output, "w")
    for word in doc:
        print(word.text, word.orth, word.lower, word.tag_, word.ent_type_, word.ent_iob)
        line = word.text + "\t" + word.ent_type_ + "\n"
        testfilespacygermeval.write(line)
        i += 1
    print(i)

    if model_dir is not None:
        save_model(ner, model_dir)


if __name__ == '__main__':

    main('spacy_trained_with_germeval_tested_on_germeval',
         'corpora/germaner/NER-de-train.tsv',
         'corpora/germaner/NER-de-test.tsv',
         'germeval',
         'germeval',
         'classifiers/spacy/spacy_germeval_trained')

    main('spacy_trained_with_germeval_tested_on_conlltesta',
         'corpora/germaner/NER-de-train.tsv',
         'corpora/conll2003/deuutf.testa',
         'germeval',
         'conll',
         'classifiers/spacy/spacy_germeval_trained')

    main('spacy_trained_with_germeval_tested_on_europarl',
         'corpora/germaner/NER-de-train.tsv',
         'corpora/europarl/ep-96-04-15-just-tokens.tsv',
         'germeval',
         'conll',
         'classifiers/spacy/spacy_germeval_trained')

    main('spacy_trained_with_conll_tested_on_germeval',
         'corpora/conll2003/deuutf.train',
         'corpora/germaner/NER-de-test.tsv',
         'conll',
         'germeval',
         'classifiers/spacy/spacy_conll_trained')

    main('spacy_trained_with_conll_tested_on_conlltesta',
         'corpora/conll2003/deuutf.train',
         'corpora/conll2003/deuutf.testa',
         'conll',
         'conll',
         'classifiers/spacy/spacy_conll_trained')

    main('spacy_trained_with_conll_tested_on_europarl',
         'corpora/conll2003/deuutf.train',
         'corpora/europarl/ep-96-04-15-just-tokens.tsv',
         'conll',
         'conll',
         'classifiers/spacy/spacy_conll_trained')

def postprocess(tagged_files):
    i = 1
    output = tagged_files + ".tsv"
    with open(tagged_files, "r") as pre, open(output, "w") as post:
        for line in pre:
            line = line.split("\t")
            # Fixing ignored tokens in germaner conll formated files by stanford ner on lines 64899 and 99279
            #if i == 64899 or i == 99279:
            #    post.write("<>" + "\t" + "O" + "\n")
            if len(line) >= 1:
                if line[0] == "####":
                    post.write("")
                elif line[0] == " ":
                    post.write("\n")
                else:
                    post.write(line[0] + "\t" + line[1])
            else:
                print(line, i)
            i += 1

postprocess("spacy_trained_with_conll_tested_on_conlltesta")
postprocess("spacy_trained_with_conll_tested_on_germeval")
postprocess("spacy_trained_with_conll_tested_on_europarl")

postprocess("spacy_trained_with_germeval_tested_on_conlltesta")
postprocess("spacy_trained_with_germeval_tested_on_germeval")
postprocess("spacy_trained_with_germeval_tested_on_europarl")
