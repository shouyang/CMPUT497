Authors:
    Shouyang Zhou (ccid: shouyang)

Contents:
    - Root: Various python scripts used. Details to follow below.
    - A3DataCleaned: Original datasets of the assignment.
    - AdjData: Adjusted data files consumed by scripts.
    - Part 2 && Part 3: Folders containing evaluation and data analysis for parts of the report.
    - Stanford Post Tagger: Java files required for the tagger and my taggers:
        - My taggers :
            - Stanford1, Stanford2: Domain models
            - ELL: Ell models

Scripts:
    - All scripts work on command line arguments.
    - All scripts also just print to stdoout, I did a lot of redirecting into text files.

    - nltk prefixed {brill, hmm, brill_baseline}:
        - These take two file paths, they train on one and tag the other
            - First filepath: Training file, should in the A3DataCleaned tabluar format
            - Second filepath: Tagging file, should be in the adjusted data format, ie underscore suffixed tags one sentence per line.

    - test_file_adjuster / train_file_adjuster:
        - Both take a single command line argument specifing the input file, should be in A3DataCleaned format.
        - The train file adjuster generates something like:
            - "Hello_NN World_JJ Something_VB Else_NPS"
            - "Sentences_NN On_JJ New_VB Lines_NPS"

        - The test file does the same just without the suffixes.
        - The python scripts actually train using the A3DataCleaned format.
            - Only the Stanford POS tagger needs the train_file_adjuster format.

    - Tabluate Results:
        - This generates a five column .tsv of matching words in the "truth" text to that of the predicted tagged text.
        - Takes three arguments:
            - First filepath: Truth file, in A3DataCleaned e.g. ELLTest.text
            - Second filepath: Estimated tagged file e.g. "./Part2/results/stanford_Domain2_on_domain2.txt"
            - Third filepath: file used to train the model used e.g. "Domain2Train.txt"

            - Example: python .\tabluate_results.py .\A3DataCleaned\ELLTest.txt .\stanfordELL__ELL.txt .\A3DataCleaned\ELLTrain.txt > stanford_ELL_ELL.csv

How to Train / Test:
    - For the stanford pos tagger:
        - Make an adjusted training file via the train_file_adjuster
        - Create a prop file like the ones in the stanford tagger folder, editing the training filepath.
        - Use the standford tagger jar as directed by the author's website, passing the new prop file.
        - Example: java -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -props Stanford1.tagger.props


        - Test by passing in one of the test files and the new model.
        - Example: java -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model .\ELL.tagger -textFile ELL_test_file.txt -outputFile ELL_tagger__ELL.txt

    - Python scripts:
        - The python scripts just take the two files as required in the section above.
            - Training files, don't need any adjustment
            - Testing files do, see the files suffixed _test_file.
            - Example: python .\nltk_hmm_replacement.py .\A3DataCleaned\Domain1Train.txt .\AdjData\ELL_test_file.txt > hmmReplacement_Domain1_ELL.txt

References:
    - https://stackoverflow.com/questions/34692987/cant-make-stanford-pos-tagger-working-in-nltk
    - https://www.geeksforgeeks.org/nlp-brill-tagger/
    - https://gist.github.com/blumonkey/007955ec2f67119e0909
    - https://nlp.stanford.edu/software/pos-tagger-faq.html#distsim