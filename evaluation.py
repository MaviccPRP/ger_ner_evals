# Class to evaluate tsv annotated ner files with gold standard file
from sklearn.metrics import f1_score
import os, re


class evaluation(object):
    '''
    # Class to evaluate tsv annotated ner files with gold standard file
    '''

    def evaluate(self):
        classified_output = "classified_output/"
        for file in os.listdir(classified_output):
            filename = os.fsdecode(file)
            if filename.endswith(".tsv"):
                p = re.findall('_[a-z]+[.-]+', filename)
                if '_germeval.' in p or '_germeval-' in p:
                    print(filename)
                    self.evaluate_('corpora/germaner/NER-de-test-conll-formated.txt',
                                  classified_output + filename)
                elif '_conlltesta.' in p or '_conlltesta-' in p:
                    print(filename)
                    self.evaluate_('corpora/conll2003/deuutf.testa',
                                 classified_output+filename)
                elif '_europarl.' in p or '_europarl-' in p:
                    print(filename)
                    self.evaluate_('corpora/europarl/ep-96-04-15_pado_annotated.tsv',
                                  classified_output + filename)

    def evaluate_(self, gold_anno, anno_file):
        '''
        Evaluate the annotated files.
        Prints the F1-Score for every annotated file
        '''
        anno_gold = gold_anno
        anno_ne = []
        gold_ne = []

        with open(anno_file, "r") as anno, open(anno_gold, "r") as gold:
            for linea, lineg in zip(anno, gold):
                linea = linea.split()
                lineg = lineg.split()

                if len(linea) > 1:
                    # Add annotated nes
                    if "OTH" in linea[-1]:
                        anno_ne.append("OTH")
                    elif "MISC" in linea[-1]:
                        anno_ne.append("OTH")
                    elif "PER" in linea[-1]:
                        anno_ne.append("PER")
                    elif "LOC" in linea[-1]:
                        anno_ne.append("LOC")
                    elif "ORG" in linea[-1]:
                        anno_ne.append("ORG")
                    elif linea[-1] == "O":
                        anno_ne.append("O")

                # spacy didnt annotate anything
                elif len(linea) == 1:
                    anno_ne.append("O")

                # Add gold nes
                if len(lineg) > 1:
                    if "OTH" in lineg[-1]:
                        gold_ne.append("OTH")
                    elif "MISC" in lineg[-1]:
                        gold_ne.append("OTH")
                    elif "PER" in lineg[-1]:
                        gold_ne.append("PER")
                    elif "LOC" in lineg[-1]:
                        gold_ne.append("LOC")
                    elif "ORG" in lineg[-1]:
                        gold_ne.append("ORG")
                    elif lineg[-1] == "O":
                        gold_ne.append("O")

            print("F1-Score:\t", f1_score(gold_ne, anno_ne, average="macro"))


if __name__ == '__main__':
    Ev = evaluation()
    print("Evaluating NER tools on CoNLL and GermEval")
    Ev.evaluate()