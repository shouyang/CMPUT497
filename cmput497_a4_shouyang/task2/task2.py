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

        self.subject  = self.raw["pair"]["subject"]
        self.subject["name"] = self.subject["name"].upper()

        self.object   = self.raw["pair"]["object"]
        self.object["name"] = self.object["name"].upper()

        self.entities  = self.gen_entities()
        self.sentence  = self.gen_adjusted_sentnece()
        self.doc       = nlp(self.sentence)

        self.subject_path, self.object_path = self.gen_paths()
        self.lca       = self.gen_lca()



    def gen_entities(self):
        entity_pattern = "\[\[ (.*?) \]\]"


        res = {}

        res["SUBJECT"] = self.subject
        res["OBJECT"]  = self.object

        entity_counter = 1
        for substring in re.findall(entity_pattern, self.raw_sentence):
            entity    = substring.split("|")[0].strip().upper()
            entity_id = substring.split("|")[1].strip()

            entity_dict = {"name": entity, "mid": entity_id}

            if entity_dict["mid"] not in [ x["mid"] for x in res.values()]:
                res["ENTITY" + str(entity_counter)] =  entity_dict
                entity_counter += 1

        return res

    def gen_adjusted_sentnece(self):
        adjusted_sentence = []

        in_brackets = False
        for word in self.raw_sentence.split(" "):
            if word == "[[":
                in_brackets = True
                continue
            if word == "]]":
                in_brackets = False
                continue

            if word.startswith("/m/"):

                for key, values in self.entities.items():
                    if word == values["mid"]:
                        adjusted_sentence.append(key)
                        break
                continue

            if in_brackets:
                continue

            adjusted_sentence.append(word)

        return " ".join(adjusted_sentence)

    def gen_paths(self):
        SUBJECT_path  = []
        OBJECT_path   = []


        subject_found = False
        object_found  = False
        for token in self.doc:

            if token.text == "SUBJECT":
                SUBJECT_path = [token] + list(token.ancestors)
                subject_found = True
            if token.text == "OBJECT":
                OBJECT_path  = [token] + list(token.ancestors)
                object_found  = True

        assert(subject_found and object_found)
        return (SUBJECT_path, OBJECT_path)

    def gen_lca(self):

        cur = "-None-"
        for s,o in zip(self.subject_path[::-1], self.object_path[::-1]):
            if s == o:
                cur = s.text
            else:
                break

        return cur

def generate_results_file(input_filepath):
    sents = []
    with open(input_filepath) as json_data:
        json_stuff = json.load(json_data)

        for json_blurb in json_stuff:
            sents.append(JSONSentence(json_blurb))

    output_file_name = input_filepath.replace(".json", ".txt")
    output_file_name = output_file_name.replace("a4_data/", "")

    with open(output_file_name, "w", encoding ="utf-8") as out_file:

        for sent in sents:

            #
            out_file.write(sent.sentence + "\n")

            #
            for key, value in sent.entities.items():
                out_file.write("{0}:\t[[ {1} | {2} ]]\n".format(key, value["name"], value["mid"]))

            #
            out_file.write("SUBJECT PATH: " + " -> ".join([ x.text for x in sent.subject_path]) + "\n")
            out_file.write("OBJECT  PATH: " + " -> ".join([ x.text for x in sent.object_path]) + "\n")
            out_file.write(sent.lca + "\n")

            out_file.write("\n\n")

    return

nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":
    """ The following will parse items from the files listing or generate
        the text files as per the assignment task 1 specifications.

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
#    sents = []
#    with open(files[1]) as json_data:
#        json_stuff = json.load(json_data)
#
#        for json_blurb in json_stuff:
#            sents.append(JSONSentence(json_blurb))
#
#
#    for idx, sent in enumerate(sents[:10]):
#
#        print(idx, sent.raw_sentence)
#        print(idx, sent.sentence)
#        print(idx, sent.entities)
#
#        print(idx, sent.subject_path)
#        print(idx, sent.object_path)
#        print(idx, sent.lca)
#
#
#        x = input()
#        if x:
#            break
#
#        else:
#            print(idx, "--")




