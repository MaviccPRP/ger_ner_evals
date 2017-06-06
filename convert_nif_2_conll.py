# Class to convert nif to conll-tsv
from rdflib import Graph, Variable
import pprint, re

class conll2nif(object):
    '''
    Class to convert nif to conll-tsv
    '''
    def __init__(self, nif_file, conll_file):
        '''
        :param nif_file:
        :param conll_file:
        '''
        self.nif_file = nif_file
        self.conll_file = conll_file

    def sparql_parse(self):
        '''
        Parse the nif files via SPARQL
        :return: the named entities including char indexes and entity class,
        return the full text from the isString attribute of the nif file.
        '''
        nif = self.nif_file
        # or, more compactly:
        def sequence_in(s1, s2):
            """Does `s1` appear in sequence in `s2`?"""
            return bool(re.search(".*".join(s1), s2))

        g = Graph()
        g.parse(nif, format="turtle")

        qres = g.query(
            """SELECT ?b ?a ?bline ?eline
               WHERE {
                  ?c nif:anchorOf ?a .
                  ?c itsrdf:taClassRef ?b .
                  ?c nif:endIndex ?eline .
                  ?c nif:beginIndex ?bline .
               } ORDER BY ASC(?eline)""")

        qfulltext = g.query(
            """SELECT ?a
               WHERE {
                ?b nif:isString ?a
               }""")

        fulltext = ""
        for row in qfulltext:
            fulltext = row[0]

        return qres, fulltext

    def convert(self):
        '''
        Convert the Sparql results into a list of named entities tab seperated with the class
        :return: list
        '''
        qres = self.sparql_parse()[0]
        fulltext = self.sparql_parse()[1]

        corpus_dict = {}
        positions = []

        for row in qres:
            #if 'Bruno' in str(row[1]):
            #    print(str(row[1]))
            string = re.sub(r'^[\W]+', '', str(row[1]))
            string = re.sub(r'[\W]+$', '', str(row[1]))
            #    print(string)
            #string = str(row[1])
            if "Person" in row[0]:
                corpus_dict[string] = ["PER", str(row[2]),str(row[3])]
            elif "Location" in row[0]:
                corpus_dict[string] = ["LOC", str(row[2]),str(row[3])]
            elif "Organization" in row[0]:
                corpus_dict[string] = ["ORG", str(row[2]),str(row[3])]
            else:
                corpus_dict[string] = ["OTH", str(row[2]),str(row[3])]

            positions.append(int(str(row[2])))
            positions.append(int(str(row[3])))

        d = 0
        word = ""
        nes = []

        for i in range(len(fulltext)):
            if i >= positions[d] and i <= positions[d+1]:
                word += fulltext[i]
            elif fulltext[i] != " ":
                word += fulltext[i]
            else:
                if word != "":
                    word = re.sub(r'^\n', '', word)
                    is_cardinal = re.match('^[0-9]+\.$', word)
                    is_abbr = re.match('[a-zA-Z]+\.[a-zA-Z]\.$', word)
                    if len(word) > 1:
                        if word[0] == "(":
                            nes.append(word[0] + "\t" + "O")
                            word = re.sub(r'^\(', '', word)
                        else:
                            pass

                        if word[0] == "\"":
                            nes.append(word[0] + "\t" + "O")
                            word = re.sub(r'^\"', '', word)
                        else:
                            pass

                        if word[-1] == ")":
                            word = re.sub(r'\)$', '', word)
                            if word[-1] == ".":
                                word = re.sub(r'\.$', '', word)
                                nes.append(word + "\t" + "O")
                                nes.append('.' + "\t" + "O")
                            else:
                                nes.append(word + "\t" + "O")
                            nes.append(')' + "\t" + "O")

                        elif word[-1] == "\"":
                            word = re.sub(r'\"$', '', word)
                            nes.append(word + "\t" + "O")
                            nes.append('\"' + "\t" + "O")

                        elif word[-1] == "." and not is_cardinal and not is_abbr:
                            word = re.sub(r'\.$', '', word)
                            nes.append(word + "\t" + "O")
                            nes.append('.' + "\t" + "O")
                            nes.append("\n" + "\t" + "O")

                        elif word[-1] == ",":
                            word = re.sub(r'\,$', '', word)
                            nes.append(word + "\t" + "O")
                            nes.append(',' + "\t" + "O")

                        elif word[-1] == ":":
                            word = re.sub(r'\:$', '', word)
                            nes.append(word + "\t" + "O")
                            nes.append(':' + "\t" + "O")

                        elif word[-1] == "!":
                            word = re.sub(r'\!$', '', word)
                            nes.append(word + "\t" + "O")
                            nes.append('!' + "\t" + "O")

                        elif word[-1] == "?":
                            word = re.sub(r'\?$', '', word)
                            nes.append(word + "\t" + "O")
                            nes.append('?' + "\t" + "O")

                        else:
                            nes.append(word + "\t"+"O")

                        word = ""

                    else:
                        nes.append(word + "\t"+"O")
                        word = ""

            if i == positions[d+1]-1 and d <= len(positions)-3:
                if len(word.split(" "))>1:
                    word = " ".join(word.split("  "))
                    word = word.strip()
                word = re.sub(r'^[\W]+', '', word)
                word = re.sub(r'[\W]+$', '', word)
                word = re.sub(r'^\n', '', word)

                nes.append(word + "\t"+str(corpus_dict[word][0]))
                word = ""
                d += 2
        print(nes)
        return nes

    def write2conll(self):
        '''
        Uses the list of convert() to write it into a tsv file
        '''
        count = 1
        conll_file = self.conll_file
        nes = self.convert()
        with open(conll_file, "w") as f:
            #f.write("# New Sentence" + "\n")
            for i in nes:
                i = i.split("\t")
                #print(i)
                if i[0] != "\n":
                    if len(i[0].split()) > 1:
                        j = i[0].split()
                        for d, token in enumerate(j):
                            if d == 0 and i[-1] != "O":
                                f.write(token + "\tB-" + i[-1] + "\n")
                                count += 1
                            elif d == 0 and i[-1] == "O":
                                f.write(token + "\t" + i[-1] + "\n")
                                count += 1
                            elif i[-1] == "O":
                                f.write(token + "\t" + i[-1] + "\n")
                                count += 1
                            elif i[-1] != "O":
                                f.write(token + "\tI-" + i[-1] + "\n")
                    else:
                        if i[-1] == "O" and i[0] != ".":
                            f.write(i[0] + "\t" + i[-1] + "\n")
                        elif i[-1] == "O" and i[0] == ".":
                            f.write(i[0] + "\t" + i[-1] + "\n\n")
                        elif i[-1] != "O":
                            f.write(i[0] + "\tB-" + i[-1] + "\n")

                else:
                    f.write(i[0])
                #    #f.write("# New Sentence" + "\n")
                #    #f.write("# New Sentence" + "\n")
                #    count = 1

if __name__ == '__main__':
    Cv = conll2nif("ep-96-04-15.nif","testing")
    Cv.write2conll()