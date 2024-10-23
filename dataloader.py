# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from config import LABEL_SET, PROMPT_SET
import json
import pandas as pd
import openai

class DataDDXPlus(object):

    def __init__(self, data_path, task) :
        self.task = task
        self.data = pd.read_csv(data_path)

    def get_data_by_task(self, task):
        self.data_task = self.data
        return self.data_task

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        self.data_task = self.get_data_by_task(task)
        content = self.data_task.iloc[idx]['Information']
        label = self.data_task.iloc[idx]['Diagnosis']
        return content, label

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_label(self):
        return LABEL_SET[self.task]
