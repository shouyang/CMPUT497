"""
Author: Shouyang Zhou (id: shouyang)
"""

import json
import nltk
import spacy
from spacy.matcher import PhraseMatcher
import re

class JSONSentence:
    """ Wrapper class that takes a JSON blob to parse via Spacy
    """

    def __init__(self, json):
        """ Generates mismatches via examining POS tags of entites.

            Input should be one of the most outer JSON blob from one of the
            assignment data files.

            This function defines the pipline of parsing the raw sentence
            into mismatched entities.

        """
        self.raw = json
        self.raw_sentence = self.raw["sentence"]
        self.sentence     = self.gen_adjusted_sentence()
        self.entities     = self.gen_entities()
        self.doc          = self.gen_doc()
        self.entity_tags  = self.match_entities()
        self.mismatched_entities = self.identify_mismatches()

    def gen_adjusted_sentence(self):
        """ Generates a cleaned sentence free of bracketted annotatons.
        """
        exclude = ["[[", "]]", "|"]

        adjusted_sentence = []
        for word in self.raw_sentence.split(" "):
            if word in exclude:
                continue
            if word.startswith("/m/"):
                continue

            adjusted_sentence.append(word)

        return " ".join(adjusted_sentence)

    def gen_entities(self):
        """ Generates all entities found in the sentence via regular expression.
        """
        entity_pattern = "\[\[ (.*?) \]\]"

        entities = []
        for substring in re.findall(entity_pattern, self.raw_sentence):

            entry = substring.split("|")[0].strip()

            if entry not in entities:
                entities.append(entry)

        return entities

    def gen_doc(self):
        return nlp(self.sentence)

    def match_entities(self):
        """ Obtains the POS tags of entities from the spacy parsed document.
        """

        matcher = PhraseMatcher(nlp.vocab)
        match_patterns = [nlp.make_doc(ent) for ent in self.entities]

        matcher.add("TerminologyList", None, *match_patterns)

        matches = matcher(self.doc)


        output = {}
        for match_id, start, end in matches:
            span = self.doc[start:end]

            entity_tags = []
            for token in span:
                entity_tags.append(token.tag_)

            if str(span) not in output:
                output[str(span)] = entity_tags

        return output

    def identify_mismatches(self):
        """ Obtains mismatched phrases by ensuring that the pos tags associated
            to each entity actually contains nouns.
        """
        must_have_tags = set(["NNP", "NNPS", "NN", "NNS"])

        assert len(self.entities) == len(self.entity_tags)

        mismatches = {}
        for phrase, tag_list in self.entity_tags.items():
            tag_set = set(tag_list)

            if tag_set.intersection(must_have_tags):
                continue
            else:
                mismatches[phrase] = tag_list

        return mismatches

def generate_results_file(input_filepath):
    sents = []
    with open(input_filepath) as json_data:
        json_stuff = json.load(json_data)

        for json_blurb in json_stuff:
            sents.append(JSONSentence(json_blurb))

    output_file_name = input_filepath.replace(".json", ".txt")
    output_file_name = output_file_name.replace("a4_data/", "")

    with open(output_file_name, "w") as out_file:
        for sent in sents:

            out_file.write(sent.raw_sentence + "\n")

            for token in sent.doc:
                out_file.write("{0}\t{1}\n".format(token,token.tag_))

            if sent.mismatched_entities:
                for key in sent.mismatched_entities:
                    out_file.write("{0}\n".format(key))

            out_file.write("\n\n")

nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    """ The following will either generate the runs files as per the assignment specifications or can be 
        used to examine the first n sentences in a filtered manner. 

        Uncomment the section desired to be ran.
    """
    files = [
        "a4_data/award.award_honor.award..award.award_honor.award_winner.json",
        "a4_data/business.business_operation.industry.json",
        "a4_data/film.actor.film..film.performance.character.json",
        "a4_data/music.artist.album.json",
        "a4_data/people.person.children.json"
    ]

    # For file generation
    for file in files:
        generate_results_file(file)


    # For viewing mismatched tags
    # sents = []
    # with open(files[4]) as json_data:
    #     json_stuff = json.load(json_data)

    #     for json_blurb in json_stuff:
    #         sents.append(JSONSentence(json_blurb))


    # for idx, sent in enumerate(sents[:100]):

    #     if sent.mismatched_entities:
    #         print(idx, sent.sentence)
    #         print(idx, sent.mismatched_entities)
    #         x = input()
    #         if x:
    #             break
    #     else:
    #         print(idx, "--")




