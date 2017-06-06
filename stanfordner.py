# This functions tags a given tsv file with named entities using the standford ner.
from nltk.tag.stanford import StanfordNERTagger

def stanford_ner_tagger(stanford_dir, jarfile, modelfile, tag_this_file, output_file):
    jarfile = stanford_dir + jarfile
    modelfile = stanford_dir + modelfile

    st = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)
    i = 0
    tagged_ne = []
    with open(tag_this_file, "r") as f:
        for line in f:
            line = line.split()
            i += 1
            if len(line) > 0:
                tagged_ne.append(line[0])
            else:
                # Remove the SENENDs from the output file afterwards. Needed to keep the format consistent
                # Keep in mind, that some "/" are still removed. Is replace in postprecessing step.
                tagged_ne.append("SENEND")
    print(tagged_ne)

    # Tag the file using Stanford NER
    out = st.tag(tagged_ne)

    # Write the results to a tsv file
    with open(output_file, "w") as f:
        for i in out:
            f.write(str(i[0])+"\t"+i[1]+"\n")

# Tagging conll with conll model
stanford_ner_tagger('classifiers/stanford-ner-2016-10-31/', \
                    'stanford-german-corenlp-2016-10-31-models.jar', \
                    "classifiers/german.conll.hgc_175m_600.crf.ser.gz", \
                    "corpora/conll2003/deuutf.testa", \
                    "stanford_trained_with_conll_tested_on_conlltesta")

# Tagging europarl with conll model
stanford_ner_tagger('classifiers/stanford-ner-2016-10-31/', \
                    'stanford-german-corenlp-2016-10-31-models.jar', \
                    "classifiers/german.conll.hgc_175m_600.crf.ser.gz", \
                    "corpora/europarl/ep-96-04-15_pado_annotated.tsv", \
                    "stanford_trained_with_conll_tested_on_europarl")

# Tagging germeval with conll model
stanford_ner_tagger('classifiers/stanford-ner-2016-10-31/', \
                    'stanford-german-corenlp-2016-10-31-models.jar', \
                    "classifiers/german.conll.hgc_175m_600.crf.ser.gz", \
                    "corpora/germaner/NER-de-test-conll-formated.txt", \
                    "stanford_trained_with_conll_tested_on_germeval")

# Tagging conll with germeval model
stanford_ner_tagger('classifiers/stanford-ner-2016-10-31/', \
                    'stanford-german-corenlp-2016-10-31-models.jar', \
                    "classifiers/ner-model-germaeval.ser.gz", \
                    "corpora/conll2003/deuutf.testa", \
                    "stanford_trained_with_germeval_tested_on_conlltesta")

# Tagging europarl with germeval model
stanford_ner_tagger('classifiers/stanford-ner-2016-10-31/', \
                    'stanford-german-corenlp-2016-10-31-models.jar', \
                    "classifiers/ner-model-germaeval.ser.gz", \
                    "corpora/europarl/ep-96-04-15_pado_annotated.tsv", \
                    "stanford_trained_with_germeval_tested_on_europarl")

# Tagging germeval with germeval model
stanford_ner_tagger('classifiers/stanford-ner-2016-10-31/', \
                    'stanford-german-corenlp-2016-10-31-models.jar', \
                    "classifiers/ner-model-germaeval.ser.gz", \
                    "corpora/germaner/NER-de-test-conll-formated.txt", \
                    "stanford_trained_with_germeval_tested_on_germeval")

def postprocess(tagged_files):
    i = 1
    output = "classified_output/" + tagged_files + ".tsv"
    with open(tagged_files, "r") as pre, open(output, "w") as post:
        for line in pre:
            line = line.split("\t")
            # Fixing ignored tokens in germaner conll formated files by stanford ner on lines 64899 and 99279
            if i == 64899 or i == 99279:
                post.write("<>" + "\t" + "O" + "\n")
            if len(line) >= 1:
                if line[0] == "SENEND":
                    post.write("\n")
                else:
                    post.write(line[0] + "\t" + line[1])
            else:
                print(line, i)
            i += 1

postprocess("stanford_trained_with_conll_tested_on_conlltesta")
postprocess("stanford_trained_with_conll_tested_on_germeval")
postprocess("stanford_trained_with_conll_tested_on_europarl")

postprocess("stanford_trained_with_germeval_tested_on_conlltesta")
postprocess("stanford_trained_with_germeval_tested_on_germeval")
postprocess("stanford_trained_with_germeval_tested_on_europarl")