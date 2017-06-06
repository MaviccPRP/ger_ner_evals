# This function starts a java subprocess and tags a file with named entities using GermaNER
import subprocess, csv

def germaner_tagger(germaner, tag_this_file, output_file):
    proc = subprocess.run(["java", "-Xmx4g", "-jar", germaner, "-t", tag_this_file, "-o", output_file])

def convert_to_germaner(input, output):
    with open(input, "r") as conll, open(output, "w") as tokenized:
        for line in conll:
            line = line.split()
            if len(line):
                tokenized.write(line[0] + "\t" + line[4] + "\n")
            else:
                tokenized.write("\n")

# Convert europarl to germaner compatible
convert_to_germaner("corpora/europarl/ep-96-04-15_pado_annotated.tsv", "corpora/europarl/ep-96-04-15_pado_annotated_germanized.tsv")

# Convert CoNLL to germaner compatible
convert_to_germaner("corpora/conll2003/deuutf.testa", "corpora/conll2003/deuutf_germanized.testa")

# Using 2 column converted version of the CoNLL 2003 data
germaner_tagger("classifiers/GermaNER/GermaNER-nofb-09-09-2015.jar", \
                "corpora/conll2003/deuutf_germanized.testa", \
                "classified_output/germaner_trained_with_germeval_tested_on_conlltesta.tsv")

germaner_tagger("classifiers/GermaNER/GermaNER-nofb-09-09-2015.jar", \
                "corpora/germaner/NER-de-test-conll-formated.txt", \
                "classified_output/germaner_trained_with_germeval_tested_on_germeval.tsv")

# Using 2 column converted version of the pado annotated europarl data
germaner_tagger("classifiers/GermaNER/GermaNER-nofb-09-09-2015.jar", \
                "corpora/europarl/ep-96-04-15_pado_annotated_germanized.tsv", \
                "classified_output/germaner_trained_with_germeval_tested_on_europarl.tsv")
