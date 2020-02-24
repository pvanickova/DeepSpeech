#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio

from util.text import Alphabet

n_input = 26
n_context = 9
n_steps = 16
n_cell_dim = 2048


def samples_to_mfccs(samples, sample_rate):
    # 16000 = default sample rate
    # 32 = default feature extraction audio window length in milliseconds
    audio_window_samples = 16000 * (32 / 1000)
    # 20 = default feature extraction window step length in milliseconds
    audio_step_samples = 16000 * (20 / 1000)
    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=audio_window_samples,
                                                  stride=audio_step_samples,
                                                  magnitude_squared=True)

    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=n_input)
    mfccs = tf.reshape(mfccs, [-1, n_input])

    return mfccs, tf.shape(input=mfccs)[0]


def audiofile_to_features(wav_filename):
    samples = tf.io.read_file(wav_filename)
    decoded = contrib_audio.decode_wav(samples, desired_channels=1)
    features, features_len = samples_to_mfccs(decoded.audio, decoded.sample_rate)
    return features, features_len


def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x, axis=1))
    return e_x / e_x.sum(axis=1)


def create_overlapping_windows(batch_x):
    batch_size = tf.shape(input=batch_x)[0]
    window_width = 2 * n_context + 1
    num_channels = n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x


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
    with tf.io.gfile.GFile(args.model_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # print(graph_def.node)


    with tf.Graph().as_default() as graph:

        tf.import_graph_def(graph_def, name="prefix")

        # currently hardcoded values used during inference

        with tf.compat.v1.Session(graph=graph) as session:

            features, features_len = audiofile_to_features(args.input_file_path)
            previous_state_c = np.zeros([1, n_cell_dim ])
            previous_state_h = np.zeros([1, n_cell_dim])

            # Add batch dimension
            features = tf.expand_dims(features, 0)
            features_len = tf.expand_dims(features_len, 0)

            # Evaluate
            features = create_overlapping_windows(features).eval(session=session)
            features_len = features_len.eval(session=session)

            # we are interested only into logits, not CTC decoding
            inputs = {'input': graph.get_tensor_by_name('prefix/input_node:0'),
                      'previous_state_c': graph.get_tensor_by_name('prefix/previous_state_c:0'),
                      'previous_state_h': graph.get_tensor_by_name('prefix/previous_state_h: 0'),
                      'input_lengths': graph.get_tensor_by_name('prefix/input_lengths:0')}
            outputs = {'outputs': graph.get_tensor_by_name('prefix/raw_logits:0'),
                       'new_state_c': graph.get_tensor_by_name('prefix/new_state_c:0'),
                       'new_state_h': graph.get_tensor_by_name('prefix/new_state_h: 0'),
                       }

            logits = np.empty([0, 1, alphabet.size() + 1])

            # the frozen model only accepts input split to 16 step chunks,
            # if the inference was run from checkpoint instead (as in single inference in deepspeech script), this loop wouldn't be needed
            for i in range(0, features_len[0], n_steps):
                chunk = features[:, i:i + n_steps, :, :]
                chunk_length = chunk.shape[1];
                # pad with zeros if not enough steps (len(features) % FLAGS.n_steps != 0)
                if chunk_length < n_steps:
                    chunk = np.pad(chunk,
                                   (
                                       (0, 0),
                                       (0, n_steps - chunk_length),
                                       (0, 0),
                                       (0, 0)
                                   ),
                                   mode='constant',
                                   constant_values=0)

                # need to update the states with each loop iteration
                logits_step, previous_state_c, previous_state_h = session.run([outputs['outputs'], outputs['new_state_c'], outputs['new_state_h']],feed_dict={
                    inputs['input']: chunk,
                    inputs['input_lengths']: [chunk_length],
                    inputs['previous_state_c']: previous_state_c,
                    inputs['previous_state_h']: previous_state_h,
                })

                logits = np.concatenate((logits, logits_step))

            logits = np.squeeze(logits)

            row_output = []
            for j in range(args.predicted_character_count):
                row_output.append([])

            # now sort logits and turn them into characters + probabilities
            for i in range(0, len(logits)):
                softmax_output = softmax(logits[i])
                indexes_sorted = softmax_output.argsort()[args.predicted_character_count * -1:][::-1]
                most_likely_chars = ''
                chars_probability = ''
                for j in range(args.predicted_character_count):
                    char_index = indexes_sorted[j]
                    if char_index < alphabet.size():
                        text = alphabet._string_from_label(char_index)
                        most_likely_chars += text+' '
                        row_output[j].append(text)
                        chars_probability += ' (' + str(softmax_output[char_index]) + ')'
                    else:
                        most_likely_chars += '- '
                        row_output[j].append('-')
                        chars_probability += ' (' + str(softmax_output[char_index]) + ')'
                print(most_likely_chars, " ", chars_probability)

            with open(args.input_file_path+"_acoustic.txt","w") as out:
                for j in range(len(row_output)):
                    out.write(', '.join(row_output[j])+"\n")
                    print(row_output[j])


if __name__ == '__main__':
    run_inference()

