# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter


class Dataset(object):
    """
    This module implements the APIs for loading and using RACE dataset
    """
    def __init__(self, train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += json.load(open(train_file))
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += json.load(open(dev_file))
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += json.load(open(test_file))
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'article_token_ids': [],
                      'article_length': [],
                      'option0_ids': [],
                      'option1_ids': [],
                      'option2_ids': [],
                      'option3_ids': [],
                      'option_length': [],
                      'answer': []
                      }
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['article_token_ids'].append(sample['article'])
            batch_data['article_length'].append(len(sample['article']))
            batch_data['question_token_ids'].append(sample['questions'])
            batch_data['question_length'].append(len(sample['questions']))
            batch_data['option0_ids'].append(sample['option0'])
            option_max_length = len(sample['option0'])
            batch_data['option1_ids'].append(sample['option1'])
            option_max_length = max(option_max_length, len(sample['option1']))
            batch_data['option2_ids'].append(sample['option2'])
            option_max_length = max(option_max_length, len(sample['option2']))
            batch_data['option3_ids'].append(sample['option3'])
            batch_data['option_length'].append(max(option_max_length, len(sample['option3'])))
            if sample['answer'] == 'A':
                batch_data['answer'].append([1, 0, 0, 0])
            if sample['answer'] == 'B':
                batch_data['answer'].append([0, 1, 0, 0])
            if sample['answer'] == 'C':
                batch_data['answer'].append([0, 0, 1, 0])
            if sample['answer'] == 'D':
                batch_data['answer'].append([0, 0, 0, 1])


        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = max(batch_data['question_length'])
        pad_q_len = max(batch_data['option_length'])
        #print(batch_data['question_token_ids'])
        batch_data['article_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                            for ids in batch_data['article_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['question_token_ids']]

        batch_data['option0_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['option0_ids']]
        batch_data['option1_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                     for ids in batch_data['option1_ids']]
        batch_data['option2_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                     for ids in batch_data['option2_ids']]
        batch_data['option3_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                     for ids in batch_data['option3_ids']]

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['article']:
                    yield token
                for token in sample['questions']:
                    yield token
                for token in sample['option0']:
                    yield token
                for token in sample['option1']:
                    yield token
                for token in sample['option2']:
                    yield token
                for token in sample['option3']:
                    yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['article'] = vocab.convert_to_ids(sample['article'])
                sample['questions'] = vocab.convert_to_ids(sample['questions'])
                sample['option0'] = vocab.convert_to_ids(sample['option0'])
                sample['option1'] = vocab.convert_to_ids(sample['option1'])
                sample['option2'] = vocab.convert_to_ids(sample['option2'])
                sample['option3'] = vocab.convert_to_ids(sample['option3'])


    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))

        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
