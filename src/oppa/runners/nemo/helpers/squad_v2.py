""" 
This is the same as NeMo's SquadDataModule but changed squad to squad_v2
"""


from nemo.collections import llm

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datasets import DatasetDict, load_dataset

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


class SquadV2DataModule(FineTuningDataModule, IOMixin):

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = False,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        super().__init__(
            dataset_root=get_dataset_root("GEM/squad_v2"),
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            packed_sequence_specs=packed_sequence_specs,
            dataset_kwargs=dataset_kwargs,
        )

    def prepare_data(self) -> None:
        # if train file is specified, no need to do anything
        if not self.train_path.exists() or self.force_redownload:
            dset = self._download_data()
            self._preprocess_and_split_data(dset)
        super().prepare_data()

    def _download_data(self):
        logging.info(f"Downloading {self.__class__.__name__}...")
        return load_dataset(
            "GEM/squad_v2",
            cache_dir=str(self.dataset_root),
            download_mode="force_redownload" if self.force_redownload else None,
            trust_remote_code=True
        )

    def _preprocess_and_split_data(
        self, dset: DatasetDict, split_val_from_train: bool = True, val_proportion: float = 0.05
    ):
        """Preprocesses and splits the downloaded dataset into training, validation, and test sets.

        Args:
            dset (DatasetDict): The downloaded dataset object.
            split_val_from_train (bool, optional): Whether to split the validation set from the training set.
                If False, the validation set is split from the test set. Defaults to True.
            val_proportion (float, optional): The proportion of the training or test set to be used for the validation split.
                Defaults to 0.05.
        """
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")
        save_splits = {}
        train_set = dset.get('train')
        val_set = dset.get('validation')
        if split_val_from_train:
            split_dataset = train_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = split_dataset['train']
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = val_set
        else:
            split_dataset = val_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = train_set
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = split_dataset['train']

        for split_name, dataset in save_splits.items():
            output_file = self.dataset_root / f"{split_name}.jsonl"

            with output_file.open("w", encoding="utf-8") as f:
                for example in dataset:
                    json_line = {}
                    # Write each example as a JSON line in the output file
                    json_line["input"] = (
                        "Context: " + example["context"] + " Question: " + example['question'] + " Answer:"
                    )
                    if len(example["answers"]["text"]) > 0:
                        json_line["output"] = example["answers"]["text"][0]
                    else:
                        # for cases where there are no answers
                        json_line["output"] = ""
                    if split_name == "test":
                        json_line["original_answers"] = example["answers"]["text"]
                    f.write(json.dumps(json_line) + "\n")

            logging.info(f"{split_name} split saved to {output_file}")

        if self.delete_raw:
            for p in self.dataset_root.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                elif '.jsonl' not in str(p.name):
                    p.unlink()

    def reconfigure_limit_batches(self):
        return



def make_squad_v2_hf_dataset(tokenizer, seq_length, global_batch_size, micro_batch_size, seed):
    
    # def formatting_prompts_func(example):
    #     formatted_text = [
    #         f"Context: {example['context']} Question: {example['question']} Answer:",
    #         f" {example['answers']['text'][0].strip()}",
    #     ]
    #     context_ids, answer_ids = list(map(tokenizer.text_to_ids, formatted_text))
    #     if len(context_ids) > 0 and context_ids[0] != tokenizer.bos_id:
    #         context_ids.insert(0, tokenizer.bos_id)
    #     if len(answer_ids) > 0 and answer_ids[-1] != tokenizer.eos_id:
    #         answer_ids.append(tokenizer.eos_id)

    #     return dict(
    #         labels=(context_ids + answer_ids)[1:],
    #         input_ids=(context_ids + answer_ids)[:-1],
    #         loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
    #     )

    # datamodule = llm.HFDatasetDataModule(
    #     'squad_v2',
    #     split='train', 
    #     pad_token_id=tokenizer.eos_id,
    #     seq_length=seq_length, 
    #     global_batch_size=global_batch_size, 
    #     micro_batch_size=micro_batch_size,
    # )
    # datamodule.map(
    #     formatting_prompts_func,
    #     batched=False,
    #     batch_size=2,
    #     remove_columns=["id", "title", "context", "question", 'answers'],
    # )
    
    datamodule = SquadV2DataModule(
        seq_length=seq_length, 
        tokenizer=tokenizer,
        global_batch_size=global_batch_size, 
        micro_batch_size=micro_batch_size,
        seed=seed,
    )
    return datamodule