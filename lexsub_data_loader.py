import logging
from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

import random


from utils import split_line, strip_accents

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

# Links to basic supported lexical substitution data sets.
LEXSUB_DATASET_DRIVE_URLS = {
    "coinco": "https://docs.google.com/uc?export=download&id=1Sb7I_0NpBJNq4AvMyAc9HJZidamJm-Rx",
    "semeval_all": "https://docs.google.com/uc?export=download&id=1TG-B09n2K5oRd_tJzMlBNhe0Jr_89s5c",
    "semeval_test": "https://docs.google.com/uc?export=download&id=1StQwn2d1eYy3phHfWqAyRYE7CTLsO2pg",
    "semeval_trial": "https://docs.google.com/uc?export=download&id=1SiPovrnD_EMrdhkyII3Vkw-jinUZZBqn",
    "twsi2": "https://docs.google.com/uc?export=download&id=1SYljWOOlkIPfcc8GWlm_ioVW9n__dZ83",
}

# List of supported data sets.
LEXSUB_DATASETS = ("semeval_all", "semeval_trial", "semeval_test", "coinco", "twsi2")


class My_data_set(data.Dataset):
    DATA_COLUMNS = [
        "context",
        "candidates",
        "target_position",
        "target_lemma",
        "pos_tag",
        "gold_subst",
        # "gold_subst_weights",
    ]

    def __init__(
        self,
        dataset_name: str,
        data_root_path: Union[str, Path],
        tokenizer,
        url: Optional[str] = None,
        with_pos_tag: bool = True,
        sent_max_length: int = 80,
        candidate_max_length: int = 10,
    ):
        """
        Reader for Lexical Substitution datasets.
        Args:
            dataset_name: Alias for dataset naming.
            data_root_path: Path for all available datasets. Datasets will be downloaded to this directory.
            url: Link for downloading dataset.
            with_pos_tag: Bool flag. If True, then the reader expects the presence of POS-tags in the dataset.
            tokenizer: RobertaTokenizerFast from huggingface transformers
        """
        # if url is None and dataset_name in LEXSUB_DATASETS:
        #     url = LEXSUB_DATASET_DRIVE_URLS[dataset_name]
        self.dataset_path = Path(data_root_path) / dataset_name
        self.tokenizer = tokenizer
        # self.url = url
        # if not self.dataset_path.exists():
        #     download_dataset(self.url, self.dataset_path)

        self.with_pos_tag = with_pos_tag
        self.sent_max_length = sent_max_length
        self.candidate_max_length = candidate_max_length

        self.dataset = self.read_dataset()


    def read_file(
        self, file_path: Union[str, Path], accents: bool = False, lower: bool = False
    ) -> List[str]:
        file_path = Path(file_path)
        if not file_path.exists():
            if self.url is None:
                raise FileNotFoundError(f"File {file_path} doesn't exist!")
            download_dataset(self.url, self.dataset_path)

        logger.info(msg=f"Reading data from {file_path} file...")
        with file_path.open("r") as f:
            data = f.readlines()

        while "\n" in data:
            data.remove("\n")
        if accents:
            data = [strip_accents(line) for line in data]
        if lower:
            data = [line.lower() for line in data]
        logger.info(msg=f"Done. File contains {len(data)} lines")
        return data


    def read_dataset(self):
        """
        Lexical Substitution dataset consists of 3 different files:
            1. sentences - file with contexts, target word positions and POS-tags.
            2. golds - file with gold substitutes and annotators info.
            3. candidates - file with candidates for Candidate Ranking task.
        """
        golds_data = self._preprocess_gold_part(
            self.read_file(self.dataset_path / "gold")
        )
        sentences_data = self._preprocess_sentence_part(
            self.read_file(self.dataset_path / "sentences")
        )
        candidates_data = self._preprocess_candidate_part(
            self.read_file(self.dataset_path / "candidates")
        )

        # Reading mapping from target to candidates
        lemma_to_candidates = {}
        for lemma, *candidates in candidates_data:
            lemma_to_candidates[lemma] = list(sorted(set(candidates)))

        # Reading golds
        golds_map = {}
        for datum in golds_data:
            gold_id = datum[1]
            assert gold_id not in golds_map, "Duplicated gold id occurred!"
            # some substitutes are multi-word expressions
            # each substitute needs to be an array, most of them are single-value
            substitutes = [pair[0].split() for pair in datum[2:] if pair]
            # gold_weights = [float(pair[1]) for pair in datum[2:] if pair]
            golds_map[gold_id] = {
                "gold_subst": substitutes,
                # "gold_subst_weights": gold_weights,
            }

        # Reading context and creating dataset
        dataset = {column: [] for column in self.DATA_COLUMNS}
        candidates = []
        context = []
        gold_subst = []
        target_pos = []
        for datum in sentences_data:
            context_id = datum[1]
            if context_id not in golds_map:
                logger.warning(f"Missing golds for context with id {context_id}")
                continue

            target, pos_tag = datum[0].split(".", maxsplit=1)
            cands = lemma_to_candidates[target + "." + pos_tag.split(".")[0]]
            # Similar to gold_subst, candidates should be an array of array
            temp_candidates = [cand.split(" ") for cand in cands]
            if len(temp_candidates) <= 1:
                continue
            else:
                candidates.append(temp_candidates)
            # else:
            #     target = datum[0]
            #     dataset["target_lemma"].append(target)
            #     dataset["pos_tag"].append(None)
            #     dataset["candidates"].append(lemma_to_candidates[target])
            dataset["target_lemma"].append(target)
            dataset["pos_tag"].append(pos_tag)
            dataset["target_position"].append(int(datum[2]))
            # dataset["context"].append(datum[3].split())
            context.append(datum[3].split())
            gold_data = golds_map[context_id]
            # dataset["gold_subst"].append(gold_data["gold_subst"])
            gold_subst.append(gold_data["gold_subst"])
            # dataset["gold_subst_weights"].append(gold_data["gold_subst_weights"])
        
        
        # for instance in context+candidates:
        #     for word in instance:
        #         self.vocabulary.add_word(word)

        tokenizer_vocab = list(set(self.tokenizer.vocab.keys()))
        for idx in range(len(context)):
            dataset["context"].append(self.tokenizer.convert_tokens_to_ids(context[idx]))
            if gold_subst[idx][0] not in candidates[idx]:
                candidates[idx].insert(0, gold_subst[idx][0])
            dataset["candidates"].append([self.tokenizer.convert_tokens_to_ids(c) for c in candidates[idx]])
            # tokenized_candidates = []
            # for candidate in candidates[idx][:self.candidate_max_length]:
            #     temp_token = self.tokenizer.convert_tokens_to_ids(candidate)
            #     if temp_token[0] == self.tokenizer.unk_token_id:
            #         self.tokenizer.add_tokens(candidate)
            #     tokenized_candidates.append(self.tokenizer.convert_tokens_to_ids(candidate))
            # dataset["candidates"].append(tokenized_candidates)
            dataset["gold_subst"].append(self.tokenizer.convert_tokens_to_ids(gold_subst[idx][0]))
        assert dataset["target_position"][-1] <= len(dataset["context"][-1]), \
                f"Wrong target position ({dataset['target_position']} in context with id {context_id})"

        return dataset


    @staticmethod
    def _preprocess_sentence_part(sentences: List[str]):
        """
        Method for processing raw lines from file with sentences.

        Args:
            sentences: List of raw lines.
        Returns:
            sentences: List of processed sentences.
        """
        for idx in range(len(sentences)):
            sentence_info = split_line(sentences[idx], sep="\t")
            sentences[idx] = sentence_info
        return sentences

    @staticmethod
    def _preprocess_candidate_part(candidates):
        """
        Method for processing raw lines from file with candidates.

        Args:
            candidates: List of raw lines.
        Returns:
            candidates: List of processed candidates.
        """
        for idx in range(len(candidates)):
            candidates_info = split_line(candidates[idx], sep="::")
            candidates[idx] = [candidates_info[0].strip()]
            candidates[idx] += candidates_info[1].split(";")
            for jdx in range(1, len(candidates[idx])):
                candidates[idx][jdx] = candidates[idx][jdx].strip()
        return candidates

    @staticmethod
    def _preprocess_gold_part(golds):
        """
        Method for processing raw lines from file with golds.

        Args:
            golds: List of raw lines.
        Returns:
            golds: List of processed golds.
        """
        for idx in range(len(golds)):
            gold_info = split_line(golds[idx], sep="::")
            golds[idx] = gold_info[0].rsplit(maxsplit=1)
            golds[idx].extend([
                tuple(subst.strip().rsplit(maxsplit=1))
                for subst in gold_info[1].split(";")
                if subst
            ])
        return golds


    def __getitem__(self, index):
        # data_item = eval(self.raw_data[index])
        init_sent = self.dataset['context'][index][:self.sent_max_length]
        init_sent = [self.tokenizer.bos_token_id] + init_sent + [self.tokenizer.eos_token_id]

        subst_label = [0]*(self.candidate_max_length)
        

        # Take the substitute with highest weight as golden one
        gold_sub = self.dataset['gold_subst'][index]
        # randomly select candidate_max_length substs
        cand_len = len(self.dataset['candidates'][index])
        if cand_len < self.candidate_max_length:
            subst = random.sample(self.dataset['candidates'][index] + [gold_sub]*(self.candidate_max_length-cand_len) , self.candidate_max_length)
        else:
            subst = random.sample(self.dataset['candidates'][index], self.candidate_max_length)
        if gold_sub not in subst:
            subst[random.randint(0,len(subst)-1)] = gold_sub
        subst_label[subst.index(gold_sub)] = 1

        # Have multiple right word with different weight
        # set the prompt_label as weights instead of True/False
        # for sub in self.dataset['gold_subst'][index]:
        #     if sub in subst:
                # right_subst_index.append(subst.index(sub))
        # subst_weights = [0]*(self.candidate_max_length)
        # for gold_idx, right_idx in right_subst_index:
            # subst_weights[right_idx] = self.dataset['gold_subst_weights'][index][gold_idx]

        # subst_word_index = [len(init_sent) + i for i in range(len(subst))]

        prompt = []
        subst_word_index = []
        for s in subst:
            tmp = range(len(init_sent)+len(prompt), len(init_sent)+len(prompt)+len(s))
            subst_word_index.append(list(tmp))
            prompt.extend(s)
            prompt.append(self.tokenizer.eos_token_id)

        final_sent = init_sent + prompt
        while len(final_sent) < (self.sent_max_length+self.candidate_max_length):
            final_sent.append(self.tokenizer.pad_token_id)
            
        new_data_item = {}
        new_data_item['sent'] = torch.LongTensor(final_sent)
        new_data_item['candidates'] = subst #self.dataset['candidates'][index][:self.candidate_max_length]
        new_data_item['target_position'] =  self.dataset['target_position'][index]
        new_data_item['pos_tag'] = self.dataset['pos_tag'][index]
        # should lemma be a seperate dictionary?
        new_data_item['subst_S_index'] = len(init_sent)
        new_data_item['subst_label'] = subst_label
        new_data_item['subst_word_index'] = subst_word_index
        
        return new_data_item    

    def __len__(self):
        return len(self.dataset['context'])


# class Vocabulary(object):
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = []

#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         return self.word2idx[word]

#     def __len__(self):
#         return len(self.idx2word)


def batch_convert_ids_to_tensors(batch_token_ids, ignore_index):
    bz = len(batch_token_ids)
    batch_tensors = [batch_token_ids[i].squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=ignore_index).long()
    return batch_tensors

def collate_fn(data):
    batch_data = {'sent': [], 'candidates':[], 'target_position':[], "pos_tag":[], 'subst_S_index': [], 'subst_label':[], 'subst_word_index':[] }

    for data_item in data:
        for k, v in batch_data.items():
            batch_data[k].append(data_item[k])
            
    batch_data['sent'] =  batch_convert_ids_to_tensors(batch_data['sent'], ignore_index=1)  # cotext + candidates
    #batch_data['metaphor_label'] =  batch_convert_ids_to_tensors(batch_data['metaphor_label'], ignore_index=2)  # 0=no, 1=yes, 2=pad
    batch_data['candidates'] = batch_data['candidates']
    batch_data['target_position'] =  batch_data['target_position']
    batch_data['pos_tag'] =  batch_data['pos_tag']
    batch_data['subst_S_index'] =  torch.LongTensor(batch_data['subst_S_index']) # start index of candidates
    batch_data['subst_label'] =  torch.FloatTensor(batch_data['subst_label']) 
    batch_data['subst_word_index'] =  batch_data['subst_word_index'] # index of each candidates in final_sent
    return batch_data


def get_lexsub_loader(dataset_names, data_root_path, batch_size, tokenizer, candidate_max_length=50, num_workers=0):
    if len(dataset_names) == 1:
        dataset = My_data_set(dataset_names[0], data_root_path, tokenizer=tokenizer, candidate_max_length=candidate_max_length)
        generator = torch.Generator().manual_seed(42)
        train, test = data.random_split(dataset, [0.7, 0.3], generator=generator)
    else:
        # datasets = [My_data_set(name, data_root_path, tokenizer=tokenizer, candidate_max_length=candidate_max_length) for name in dataset_names]
        train = []
        test = []
        for name in dataset_names:
            dataset = My_data_set(name, data_root_path, tokenizer=tokenizer, candidate_max_length=candidate_max_length)
            generator = torch.Generator().manual_seed(42)
            tmp_train, tmp_test = data.random_split(dataset, [0.7, 0.3], generator=generator)
            train.append(tmp_train)
            test.append(tmp_train)
        train = data.ConcatDataset(train)
        test = data.ConcatDataset(test)
    train_loader = data.DataLoader(dataset=train,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                 )
    test_loader = data.DataLoader(dataset=test,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                 )
    return train_loader, test_loader