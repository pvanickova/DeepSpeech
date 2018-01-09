#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tensorflow as tf

import util.audio
from util.text import Alphabet


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_inference():
    """Load frozen graph, run inference and display most likely predicted characters"""

    parser = argparse.ArgumentParser(description='Run Deepspeech inference to obtain char probabilities')
    parser.add_argument('--input-file', type=str,
                        help='Path to the wav file', action="store", dest="input_file_path")
    parser.add_argument('--alphabet-file', type=str,
                        help='Path to the alphabet.txt file', action="store", dest="alphabet_file_path")
    parser.add_argument('--model-file', type=str,
                        help='Path to the tf model file', action="store", dest="model_file_path")
    parser.add_argument('--predicted-character-count', type=int,
                        help='Number of most likely characters to be displayed', action="store",
                        dest="predicted_character_count", default=5)
    args = parser.parse_args()

    alphabet = Alphabet(os.path.abspath(args.alphabet_file_path))

    if args.predicted_character_count >= alphabet.size():
        args.predicted_character_count = alphabet.size() - 1

    # Load frozen graph from file and parse it
    with tf.gfile.GFile(args.model_file_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:

        tf.import_graph_def(graph_def, name="prefix")

        # currently hardcoded values used during inference
        n_input = 26
        n_context = 9

        with tf.Session(graph=graph) as session:
            mfcc = util.audio.audiofile_to_input_vector(args.input_file_path, n_input, n_context)

            # we are interested only into logits, not CTC decoding
            inputs = {'input': graph.get_tensor_by_name('prefix/input_node:0'),
                      'input_lengths': graph.get_tensor_by_name('prefix/input_lengths:0')}
            outputs = {'outputs': graph.get_tensor_by_name('prefix/logits:0')}

            output = session.run(outputs['outputs'], feed_dict={
                inputs['input']: [mfcc],
                inputs['input_lengths']: [len(mfcc)]
            })

            for i in range(0, len(output)):
                softmax_output = softmax(output[i][0])
                indexes_sorted = softmax_output.argsort()[args.predicted_character_count * -1:][::-1]
                most_likely_chars = ''
                chars_probability = ''
                for j in range(args.predicted_character_count):
                    char_index = indexes_sorted[j]
                    if char_index < alphabet.size():
                        text = alphabet.string_from_label(char_index)
                        most_likely_chars += text+' '
                        chars_probability += ' (' + str(softmax_output[char_index]) + ')'
                    else:
                        most_likely_chars += '- '
                        chars_probability += ' (' + str(softmax_output[char_index]) + ')'
                print(most_likely_chars, " ", chars_probability)


if __name__ == '__main__':
    run_inference()
