# Evaluation of German Named Entity Recognition Tools

*The following tools combine named entity recognition and named
entity classification. Evaluation metric
is f1-score.*

## Motivation and Goals

There are plenty of named entity recognition (NER) tools for English Corpora. In contrast, there are just a few tools for German NER.
NER for German is much more challenging than NER for English. 
As all proper nouns are capitalized, this
feature is far less effective in comparison to English NER (Faruqui/Pado 2010).

To get an overview of the current state of research the goals of this project are the following:
- research available NER projects,
- collect published evaluation scores of these NER projects (if available),
- which German NER corpora are available freely?
- choose most promising tools and train them on alternative corpora,
- evaluate the tools on an out-of-domain corpus.

Out of scope of this project is to examine the exact methods of these NER tools. 
We just distinguish between two main methods to extract named entities:
1. ML (mainly LSTM *(<a href="https://en.wikipedia.org/wiki/Long_short-term_memory">Long short-term memory</a>)*, CRF *(<a href="https://en.wikipedia.org/wiki/Conditional_random_field">Conditional Random Field</a>)*, SVM *(<a href="https://en.wikipedia.org/wiki/Support_vector_machine">Support Vector Machines</a>)*)
2. Named Entity Linking (NEL) via Wikidata, DBpedia a.o.

NEL tools are mainly focusing on multilingual NER. These tools use a standard nif format for data input and output, which makes it challenging to compare them
with non NEL tools for NER. There are currently no out-of-the-box solutions to convert the nif format to a common format like CoNLL 2003.
Rather alot of hand-crafted adaption is needed to get proper and comparable output files during evaluation.
Due to this immense extra effort, this project will primarily focus on the non-NEL tools during training and evaluation.
Felix Sasaki mentioned in this context, that 

*"tooling for machine translation training or for training of statistical
named entity recognition currently is far more efficient relying on non-linked data
representations"*

As an exception we include the Freme NER tool into our project, because it is freely available and easy to use. In addition we can show, how awkward it is, to handle the different formats of the NEL world and the non-NEL world.

## Tools:

For building our evaluation pipeline, we use the following tools:

- Python3
- rdflib

Python3 has a comfortable handling of encodings and the rdflib helps us, to read the
nif format, which is the output of the Freme NER tool.
To improve our evaluation workflow we further developed some custom classes and functions:

- evaluate.py (to evaluate NER output data with a gold standard and get f1-score)
- germaner.py (wrapper for java tool, tag a tokenized (one token per line) using the GermaNER tool)
- stanfordner.py (wrapper for java tool, tag a tokenized (one token per line) using the Stanford NER tool)
- spacyner.py (training and tagging for the spaCy tool)
- convert_conll2spacy.py (convert conll input to spaCy data)
- convert_germaeval2spacy.py (convert the germaeval input to spaCy data)
- convert_nif_2_conll.py (convert the Freme NER output nif file to a CoNLL 2003 style tab separated file. Pipeline is still buggy, alot of manual work still to be done, as special characters are sometimes not handled properly)
- lstmner.py ()just a note, that lstm uses no api, but is a command line tool)

## Data 

If we focus on the supervised learning task of NER, there are currently two data sets available:

1. CoNLL 2003 (Tjong Kim Sang/De Meulder 2003)
2. NoSta-D NE - GermEval 2014 (Bernikova 2014)

In this project we use the German CoNLL 2003 data set. The text is from the Ger-
man newspaper Frankfurter Rundshau. The training data consists of 206.931 tokens in 12.705 sentences. The test set consists of 51.943 tokens in 3.160 sentences. The NE classes are distributed as follows:

|    |LOC    | MISC   |  ORG  | PER  |
|----| ---- |:-------|-------| -----|
|Training set|4363|2288|2427|2773|
|Test set|1035|670|773|1195|

The data set contains one token per line. Sentences are seperated by empty lines. Each line contains four fields, just the last column represents the named entities in IOB format (Tjong Kim Sang/De Meulder 2003).


The GermEval 2014 data set contains text from German Wikipedia articles and onlines news texts. The training data consists of 24.000 sentences. The test set consists of 5.100 sentences. All in all the data set contains over 590.000 tokens. The NE classes are distributed as follows:

|    |LOC    | OTH   |  ORG  | PER  |
|----| ---- |:-------|-------| -----|
|Training set|12791|6986|9889|12423|
|Test set|2683|1644|2033|2609|

The data set contains one token per line. It is similar to the CoNLL 2003 format. Additionally the first column represents numbers per sentence and a commented line at the beginning of each sentence, representing the data source and data. The third columns contains the named entities in IOB format. The fourth column containing nested named entities is ignored in this task. Additional derivates of named entities are ignore, too and treated as standard IOB classes (Bernikova 2014).

As an out-of-domain data set we used texts from the european parliament annotated by Sebastian Pado following the CoNLL 2003 guidelines (https://www.nlpado.de/~sebastian/software/ner_german.shtml). 


## Experiments and Evaluation:

### NER tools

For our evaluations, we had to choose tools from all available German NER tools (see appendix).
Our tools need to fulfill the following requirements:

- freely available and open source
- well documentated
- installation is not too complicated and training new models possible in reasonable time (*still problems with GermaNER*).
- if tool is not easy to train, tool needs to offer a German model with our NE classes (MISC/OTH, PER, LOC, ORG) out of the box. 

Majority of the demands are met by the following tools:
- Standford NER (Faruqui, Padó 2010)
- GermaNER (Benikova 2015) (problems during training)
- LSTM (Lample 2016)
- SpaCy Entity recognition (no specific publication, see https://github.com/explosion/spaCy/issues/491)
- Freme NER (Dojchinovski 2017) (no training possible)

As mentioned in the motivation, we include Freme to show the effort of including NEL tools into a non-NEL evaluation.
LSTM reaches the best scores of all multilingual NER tool and is freely available. Stanford NER and GermaNER are the best performing of monolingual NER tools and are freely available and well documented.
SpaCy is a popular non-academic python NLP tool, comparing itself with state of the art NLP tools like Stanford NLP and NLTK.

### Formats 

#### Evaluation format

For our evaluation class we needed to postprocess the tagged output of our NER tools. 
For evaluation against the CoNLL 2003 test set we converted our output data from each tool into a file containing one token per line. The first line is marked by the CoNLL '-DOCSTART-' string. Sentence ends are marked by new lines. The file contains two columns, the first column contains the token, the second line contains the named entity class in IOB format.

For evaluation against the GermEval 2014 test set and the Europarl data set we converted the data in the same file format, just without the first '-DOCSTART-' line.

#### Preprocessing and postprocessing data for each tool

**StanfordNER**: In our experiments we will use the pretrained German model on the CoNLL 2003 data set and train our own model using the GermEval 2014 data set. Finally we will evaluate the two models on all three data sets, the CoNLL 2003, GermEval 2014 and the europarl data set.
To *train* the Stanford NER with the GermEval 2014 data set, we just used the third column of the training set.
For tagging we preprocessed the data, by converting the three input data sets into a python list. Sentence ends are marked by the string 'SENDEND'. The tagger outputs a list of tuples. 

`
[('Lenaustraße'.'I-LOC'),('70', 'O')]
`

This list is reformated into our standard two column tab seperated files for the evaluator class. In postprocessing we removed the 'SENTEND' strings and fixed some special character bugs during tagging.

**GermaNER**: We could not successfully train our new model with the GermaNER tool. Due to that, we just used the pretrained German model on the CoNLL 2003 data set. For the tagging process of the GermeNER tool we preprocessed the CoNLL and the europarl data sets to a two column tab seperated version. The first column contains the tokens, the second column contains the named entity classes. The GermEval test data set could be used without preprocessing.
The output is a two colum CoNLL style format, right as we use it in our evaluator class. No extra postprocessing was needed.

**LSTM**: As there is no German model available publicly available, we had to train as well a model based on the GermEval 2014 data as for the CoNLL 2003 data. For training using the GermEval 2014 data, we had to convert it into a five column style tab seperated file. Each line contains a token. The first column contains the tokens, the fifth column the NE classes. The rest v the columns are ignored. The CoNLL data could be used out of the box.
For testing the LSTM we preprocessed the CoNLL and GermEval data, by converting them into a file containing one sentence per line. The tagged output is by default in a special tokenized format. 

`
Lenaustraße__I-LOC 70__O .
`

This we converted into a our evaluator class ready two column tab seperated file format.

**SpaCy entity recognition**: The SpaCy entity recognition tool uses a different named entity class set. That is, why we need to train our own models for both data sets, the CoNLL 2003 and the GermEval 2014 data set. For training the spaCy tool we had to preprocess the CoNLL 2003 and the GermEval 2014 data set to the spaCy specific data format (see: https://spacy.io/docs/usage/entity-recognition). 
```python
 ('Das erfuhrt Sport-Bild-Online aus internen Kreisen des DSV.',
  [(0, 3, 'O'),
   (4, 11, 'O'),
   (12, 29, 'B-ORG'),
   (30, 33, 'O'),
   (34, 42, 'O'),
   (43, 50, 'O'),
   (51, 54, 'O'),
   (55, 59, 'B-ORG')]),
```
The CoNLL 2003 and the GermEval 2014 training data is converted into a list containing tuples. Every tuple contains a string, representing the sentence, and a list of tuples, containing three entries. The first and second entry inticate the position of a specific token and the last entry represents the named entity class. 
For tagging we had to postprocess the spaCy output to our evaluation two column tab seperated format.

Freme NER

The Stanford NER and the LSTM use the NE classes defined in the CoNLL 2003 shared task.
GermaNER uses the same classes, just the MISC class is replaced to OTH. 
This makes all in all four jointly used classes (MISC/OTH, PER, LOC, ORG)
GermaNER and the Stanford NER tool use the IOB format (<a href="https://en.wikipedia.org/wiki/Inside_Outside_Beginning">link</a>)
Just the LSTM NER tool by default uses an advanced IOB format, the IOBES. Here single (S) named entities and the last token (E) of a named entity are separately recognized . For the sake of somplicity we trained this tool with the simpler IOB format.
The Freme NER tool uses neither IOB nor IOBES.

Freme NER uses a normal text file as an input and returns a turtle nif file as output.
As Freme uses a custom tokenizer, which makes it very challenging, to align the output to our test corpora.
Alternatively Freme accepts turtle as input, but conversion finally had been more efficient via plain text input.
After feeding the Freme tool with the EUROPARL corpus we converted the output into CoNLL style format.
Due to the much larger test corpora from CoNLL and GermEval, we do not test them with Freme.

The german model of spaCy uses a subset of the named entities defined in the OntoNotes 5 corpus (OntoNotes Release 5.0 2012).
To fit the tool with the named entity classes used by Stanford, LSTM, GermaNER and Freme, we trained a new model, based on our four named entity classes.

### Experiments
++++++++++++++moved from data
All investigated tools had been trained on one of the above and evaluated on either the associated development or test set.
Only the Stanford NER (Faruqui/Pado 2010) had been evaluated on a manually annotated out-of-domain corpora. The corpora contains the first two sessions of the european parliament.
The first session is used for this project, as an independent evaluation data set for all tools, as well. It is not in IOB format.
For the Freme NER tool, we convert the nif output to the common tab separated CoNLL style format using our converter and manually adaptions.
++++++++++++++


For the non-NEL tools we first converted the EUROPARL text into a two column tab seperated version. Firtst column contains the token, the second one contains the NE class. The NE classes are represented in non-IOB format.







As there are considerable differences in the used metrics between the CoNLL 2003 and the GermEval 2014 shared tasks, we evaluated the tools on our own, to receive comparable results.
We use the following metrics for evaluation:
- test and gold data  uses four classes (MISC, PER, LOC, ORG),
- we use a strict evaluation metric, True Positive we just reach, if both tokens are classified exactly the same way, e.g. gold and system annotation marks a token as PER


1. For the LSTM NER tool we train a model on CoNLL 2003. In addition we trained a model using the GermEval 2014 data set.
Both models had been evaluated on the CoNLL, GermEval and EUROPARL data set.

    **LSTM**
    - trained on: CoNLL-2003 (self-trained *still training*)
        *(tag_scheme=iobes,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=100,word_lstm_dim=100,word_bidirect=True,pre_emb=,all_emb=False,cap_dim=0,crf=True,dropout=0.5,lr_method=adam)*
        - tested on: CoNLL-2003 testa (74.0%), GermaEval (54.2%), EUROPARL (64.2%)
    - trained on: GermEval 2014 (self-trained *still training*)
        *(tag_scheme=iob,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=50,word_lstm_dim=50,word_bidirect=True,pre_emb=,all_emb=False,cap_dim=0,crf=True,dropout=0.5,lr_method=adam)*
        - tested on: CoNLL-2003 testa (62.8%), GermaEval (75.8%), EUROPARL (61.8%)
        
2. For the GermaNER we use the pre-trained model, trained on the GermEval 2014 data set. In addition we trained our own model using the 
CoNLL 2003 data set. Both models we evaluated on the CoNLL testa set, GermEval test set and EUROPARL data set. 

    **GermaNER**
    - pretrained on: GermEval 2014
        - tested on: CoNLL-2003 testa (67.8%), GermaEval (75.0%), EUROPARL (68.8%)
    - trained on: CoNLL-2003 (self-trained, *still training*)
        - tested on: CoNLL-2003 testa (X), GermaEval (X), EUROPARL (X)

3. For the Stanford NER we used the pre-trained model on the CoNLL 2003 data set, using the current version of Stanford NER v. 3.7.0. In addition we trained our own model in the GermEval 2014 data set. 
Both models we evaluated on the CoNLL, GermEval and EUROPARL data set.

    **Stanford NER**
    - pretrained on: CoNLL-2003
        - tested on: CoNLL-2003 testa (**81.1%**), GermaEval (60.1%), EUROPARL (**71.9%**)
    - trained on: GermEval 2014 (self-trained)
        - tested on: CoNLL-2003 testa (61.3%), GermaEval (75.8%), EUROPARL (60.0%)
        
3. For the spaCy Entity recognition tool we trained a model on the CoNLL 2003 data set. In addition we trained a model on the GermEval 2014 data set. 
Both models we evaluated on the CoNLL, GermEval and EUROPARL data set.

    **spaCy Entity recognition**
    - trained on: CoNLL-2003
        - tested on: CoNLL-2003 testa (73%), GermaEval (59.7%), EUROPARL (64.9%)
    - trained on: GermEval 2014 (self-trained)
        - tested on: CoNLL-2003 testa (64.1%), GermaEval (71.8%), EUROPARL (64.6%)

5. As for the last tool, we did not create our own trained model for the Freme NER. This is due to the lack of documentation about training a model.
    
    **Freme NER**
    - pretrained on: 1. DBPedia, 2. via CRF based on Stanford, Data set not clear
        - tested on: EUROPARL (66.5%)
    
## Synopsis

Our evaluationis show, taht NER for German is still a challenging task. 
We collected ten different NER system for German, having a monolinguistic approach.
These tools use different learning algorithms, though conditional random fields (CRF) are by far the most common and reach the best scores.
As for the multilingual approach, we collected six different NER tools. 
As we have seen for the monolingual tools, CRF reaches the best results here, too. 
Though the tool with the highest score combines CRF with neural networks.

We chose the best performing tools, which are freely available and reach the best f1-scores.
All other tools are either not freely available or are based on rather uncommon platforms, e.g. the haskell platform. This makes
it difficult to set them up (the Sequor).

The 'gold standard' of german NER tools was and still is the Stanford NER tool (Faruqui/Pado 2010).
Similar results we get from GermaNER (Benikova 2015), a tool which is based totally on open source and freely available resources.
But as well the LSTM tool (Lample 2016) is a promising approach for german NER.
Though there had been several publications on this topic in this decade, in the official published results all tools stay beneath an f1-score of 80%.
An exception is the current Stanford NER tool, which improved its f1-score on the CoNLL training/test pair in our experiments to 81.1%.

The Stanford NER as well reached the best out-of-domain result on the EUROPARL test corpus with an f1-score of 71.92%.

NEL approaches, such as the Freme NER tool try to bypass the lack of freely available NE corpora by using huge linked data resources as wikidata or dbpedia.
Still they cannot outperform our best evaluated NER tools.

## Future Work

Another challenging tasks for NER are medical texts. Unstructured and small data sets make 
training very hard. Thats why the evaluations on an out-of-domain data set had been a main goal of
this project. Using this knowledge, could be helpful for NER for medical texts.
Thats why a future task would be to
- (train and) test the tools on medical texts,
- investigate current research projects on NER for medical texts.

To better combine the NEL and non-NEL research, it would be desirable to develop
- generic converting tools to convert nif to CoNLL style.
- extend current DEL tools, to accept pre tokenized text for training.

## Appendix

### Monolingual Approaches

| Classifier    | Trainingset   |  Testset  | F1-Score  |
| ------------- |:----------|------| -----|
| Stanford NER (Faruqui, Padó 2010) | CoNLL-2003  | CoNLL-2003 Testa| 79.8 |
| Stanford NER (Faruqui, Padó 2010) | CoNLL-2003  | CoNLL-2003 Testb | 78.2 |
| NN+STC+All Features (Reimers† 2014)   | GermEval 2014  |  GermEval 2014 (top-level NE)     |   77.1 |
| Modular Classifier (Hänig 2014)  | GermEval 2014  | GermEval 2014 Test | 79.10  |
| GermaNER (Benikova 2015) | NoSta-D NE (GermEval 2014) | NoSta-D NE (GermEval 2014)|  76.86 |
|Sequor - Perceptron(Chrupala 2010)|CoNLL-2003 + 32 million words of unlabeled text and (ii) infobox labels in German Wikipedia articles| splitted dev set | 76.60 |
|Sequor - Perceptron(Chrupala 2010)|CoNLL-2003 + 32 million words of unlabeled text and (ii) infobox labels in German Wikipedia articles| splitted test set | 74.69 |
| CRF and Linguistic Resources (Watrin 2014)  | GermEval 2014  | GermEval 2014 Dev | ~74  |
|MoSTNER (Schüller 2014)|GermEval 2014|GermEval 2014 Dev|73.5|
| NERU (Weber 2014) | GermEval 2014  | GermEval 2014 Dev | 73.26  |
| SVM (Capsamun 2014)  | GermEval 2014  | GermEval 2014 Dev | 72.63  |
|MoSTNER (Schüller 2014)|GermEval 2014|GermEval 2014 Test|71.59|
| Stanford NER (Faruqui, Padó 2010) | Trained CoNLL-2003  | EUROPARL | 65.6  |
| Adapting Data Mining (Nouvel 2014)  |  ?|  ?| 62.39 |
| Nessy: NB+Rules (Hermann 2014)  | GermEval 2014  | GermEval 2014 Dev | 60.39  |
| Nessy: NB+Rules (Hermann 2014)  | GermEval 2014  | GermEval 2014 Test | 58.78  |

### Multilingual Approaches

| Classifier    | Trainingset   |  Testset  | F1-Score  |
| ------------- |:----------|------| -----|
| LSTM-CRF (Lample 2016)   | CoNLL-2003 | CoNLL-2003  |78.76
| Semi-Supervised Features (Agerri 2017)  | GermEval 2014  | GermEval 2014 Test | 78.42  |
| Multilingual Data for NER (Faruqui 2014)| CoNLL-2003  + Dutch CoNLL 2002| CoNLL-2003 Test?| 77.37 |
| Shallow Semi-Supervised Features (Agerri 2017) | CoNLL-2003  | CoNLL-2003 Dev | 77.18  |
| Semi-Supervised Features (Agerri 2017)  | CoNLL-2003  | CoNLL-2003 Test | 76.42  |
| Hybrid Neural Networks (Shao 2016)  |  GermEval 2014 | GermEval 2014 Test | 76.12 |
| Wikipedia Entity Type Mapping for NER (Ni 2016)| Generated Entity Linked Data from Wikipedia| Gen. EL Wiki| 71.8 |

## References

<div id="refs" class="references">
<div id="ref-agerri2016robust">
<p>Agerri, Rodrigo, and German Rigau. 2016. “Robust Multilingual Named Entity Recognition with Shallow Semi-Supervised Features.” <em>Artificial Intelligence</em> 238. Elsevier: 63–82.</p>
</div>
<div id="ref-al2015polyglot">
<p>Al-Rfou, Rami, Vivek Kulkarni, Bryan Perozzi, and Steven Skiena. 2015. “Polyglot-Ner: Massive Multilingual Named Entity Recognition.” In <em>Proceedings of the 2015 Siam International Conference on Data Mining</em>, 586–94. SIAM.</p>
</div>
<div id="ref-BENIKOVA14.276">
<p>Benikova, Darina, Chris Biemann, and Marc Reznicek. 26AD. “NoSta-d Named Entity Annotation for German: Guidelines and Dataset.” In <em>Proceedings of the Ninth International Conference on Language Resources and Evaluation (Lrec’14)</em>, edited by Nicoletta Calzolari (Conference Chair), Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis. Reykjavik, Iceland: European Language Resources Association (ELRA).</p>
</div>
<div id="ref-benikova2015c">
<p>Benikova, Darina, Seid Muhie, Yimam Prabhakaran, and Santhanam Chris Biemann. 2015. “C.: GermaNER: Free Open German Named Entity Recognition Tool.” In <em>In: Proc. Gscl-2015</em>. Citeseer.</p>
</div>
<div id="ref-capsamun2014drim">
<p>Capsamun, Roman, Daria Palchik, Iryna Gontar, Marina Sedinkina, and Desislava Zhekova. 2014. “DRIM: Named Entity Recognition for German Using Support Vector Machines.” <em>Proceedings of the KONVENS GermEval Shared Task on Named Entity Recognition, Hildesheim, Germany</em>.</p>
</div>
<div id="ref-chrupala2010named">
<p>Chrupala, Grzegorz, and Dietrich Klakow. n.d. “A Named Entity Labeler for German: Exploiting Wikipedia and Distributional Clusters.” In <em>LREC</em>.</p>
</div>
<div id="ref-DOJCHINOVSKI16.578">
<p>Dojchinovski, Milan, Felix Sasaki, Tatjana Gornostaja, Sebastian Hellmann, Erik Mannens, Frank Salliau, Michele Osella, et al. 2016. “FREME: Multilingual Semantic Enrichment with Linked Data and Language Technologies.” In <em>Proceedings of the Tenth International Conference on Language Resources and Evaluation (Lrec 2016)</em>, edited by Nicoletta Calzolari (Conference Chair), Khalid Choukri, Thierry Declerck, Marko Grobelnik, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis. Paris, France: European Language Resources Association (ELRA). <a href="https://svn.aksw.org/papers/2016/LREC_FREME_Overview/public.pdf" class="uri">https://svn.aksw.org/papers/2016/LREC_FREME_Overview/public.pdf</a>.</p>
</div>
<div id="ref-faruqui2014translation">
<p>Faruqui, Manaal. 2014. “‘ Translation Can’t Change a Name’: Using Multilingual Data for Named Entity Recognition.” <em>ArXiv Preprint ArXiv:1405.0701</em>.</p>
</div>
<div id="ref-faruqui10:training">
<p>Faruqui, Manaal, and Sebastian Padó. 2010. “Training and Evaluating a German Named Entity Recognizer with Semantic Generalization.” In <em>Proceedings of Konvens 2010</em>. Saarbrücken, Germany.</p>
</div>
<div id="ref-hanig2014modular">
<p>Hänig, Christian, Stefan Bordag, and Stefan Thomas. 2014. “Modular Classifier Ensemble Architecture for Named Entity Recognition on Low Resource Systems.” In <em>Workshop Proceedings of the 12th Edition of the Konvens Conference</em>, 113–16.</p>
</div>
<div id="ref-hermann2014nessy">
<p>Hermann, Martin, Michael Hochleitner, Sarah Kellner, Simon Preissner, and Desislava Zhekova. 2014. “Nessy: A Hybrid Approach to Named Entity Recognition for German.” <em>Proceedings of the KONVENS GermEval Shared Task on Named Entity Recognition, Hildesheim, Germany</em>.</p>
</div>
<div id="ref-Jiang2016EvaluatingAC">
<p>Jiang, Ridong, Rafael E. Banchs, Haizhou Li, Ben Abacha, and Dan Roth. 2016. “Evaluating and Combining Named Entity Recognition Systems.” In.</p>
</div>
<div id="ref-kim2012multilingual">
<p>Kim, Sungchul, Kristina Toutanova, and Hwanjo Yu. 2012. “Multilingual Named Entity Recognition Using Parallel Data and Metadata from Wikipedia.” In <em>Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Long Papers-Volume 1</em>, 694–702. Association for Computational Linguistics.</p>
</div>
<div id="ref-DBLP:journals/corr/LampleBSKD16">
<p>Lample, Guillaume, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. “Neural Architectures for Named Entity Recognition.” <em>CoRR</em> abs/1603.01360. <a href="http://arxiv.org/abs/1603.01360" class="uri">http://arxiv.org/abs/1603.01360</a>.</p>
</div>
<div id="ref-negri2004using">
<p>Negri, Matteo, and Bernardo Magnini. 2004. “Using Wordnet Predicates for Multilingual Named Entity Recognition.” In <em>Proceedings of the Second Global Wordnet Conference</em>, 169–74.</p>
</div>
<div id="ref-niimproving">
<p>Ni, Jian, and Radu Florian. n.d. “Improving Multilingual Named Entity Recognition with Wikipedia Entity Type Mapping.”</p>
</div>
<div id="ref-nothman2013learning">
<p>Nothman, Joel, Nicky Ringland, Will Radford, Tara Murphy, and James R Curran. 2013. “Learning Multilingual Named Entity Recognition from Wikipedia.” <em>Artificial Intelligence</em> 194. Elsevier: 151–75.</p>
</div>
<div id="ref-nouvel2014adapting">
<p>Nouvel, Damien, and Jean-Yves Antoine. 2014. “Adapting Data Mining for German Named Entity Recognition.” In <em>Konvens’ 2014</em>, 149–53. 1.</p>
</div>
<div id="ref-prabhakarangermaner">
<p>Prabhakaran, Darina Benikova1 Seid Muhie Yimam, and Santhanam2 Chris Biemann. n.d. “GermaNER: Free Open German Named Entity Recognition Tool.”</p>
</div>
<div id="ref-ReimersEckleKohlerSchnoberetal.2014">
<p>Reimers, Nils, Judith Eckle-Kohler, Carsten Schnober, Jungi Kim, and Iryna Gurevych. 2014. “GermEval-2014: Nested Named Entity Recognition with Neural Networks.” In.</p>
</div>
<div id="ref-richman2008mining">
<p>Richman, Alexander E, and Patrick Schone. 2008. “Mining Wiki Resources for Multilingual Named Entity Recognition.” In <em>ACL</em>, 1–9.</p>
</div>
<div id="ref-rossler2002using">
<p>Rössler, Marc. 2002. “Using Markov Models for Named Entity Recognition in German Newspapers.” In <em>Proceedings of the Workshop on Machine Learning Approaches in Computational Linguistics</em>, 29–37.</p>
</div>
<div id="ref-rossler2004corpus">
<p>Rössler, Marc. 2004. “Corpus-Based Learning of Lexical Resources for German Named Entity Recognition.” In <em>LREC</em>. Citeseer.</p>
</div>
<div id="ref-schueller2014">
<p>Schüller, Peter. 2014. “MoSTNER: Morphology-Aware Split-Tag German Ner with Factorie.” In.</p>
</div>
<div id="ref-shao2016multilingual">
<p>Shao, Yan, Christian Hardmeier, and Joakim Nivre. 2016. “Multilingual Named Entity Recognition Using Hybrid Neural Networks.” In <em>The Sixth Swedish Language Technology Conference (Sltc)</em>.</p>
</div>
<div id="ref-TjongKimSang:2003:ICS:1119176.1119195">
<p>Tjong Kim Sang, Erik F., and Fien De Meulder. 2003. “Introduction to the Conll-2003 Shared Task: Language-Independent Named Entity Recognition.” In <em>Proceedings of the Seventh Conference on Natural Language Learning at Hlt-Naacl 2003 - Volume 4</em>, 142–47. CONLL ’03. Stroudsburg, PA, USA: Association for Computational Linguistics. doi:<a href="https://doi.org/10.3115/1119176.1119195">10.3115/1119176.1119195</a>.</p>
</div>
<div id="ref-watrin2014named">
<p>Watrin, Patrick, Louis De Viron, Denis Lebailly, Matthieu Constant, and Stéphanie Weiser. 2014. “Named Entity Recognition for German Using Conditional Random Fields and Linguistic Resources.” In <em>GermEval 2014 Named Entity Recognition Shared Task-Konvens 2014 Workshop</em>.</p>
</div>
<div id="ref-weber2014neru">
<p>Weber, Daniel, and Josef Pötzl. 2014. “NERU: Named Entity Recognition for German.” <em>Proceedings of the KONVENS GermEval Shared Task on Named Entity Recognition, Hildesheim, Germany</em>.</p>
</div>
</div>
