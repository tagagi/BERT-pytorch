from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    '''
     {"bert_input": bert_input,
      "bert_label": bert_label,
      "segment_label": segment_label,
      "is_next": is_next_label}

    -bert_input：一个input，包含两个句子，句子内部15%选择输入mask,每个mask按80%，10%，10%依次输入mask, wrong word, correct word的index.
                第二句子按50%选择正确的句子，50%随机选择一个错误的句子，拼成输入。
    -bert_label：按15%选择的mask，标记其正确的target_label，其他未选择均为0.
    -segment_label：标记前后句子，[0,0,0,0,1,1,1,1....]
    -is_next：标记两个句子之间的关系，label(isNotNext:0, isNext:1)

    '''

    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory: # 如果语料库长度不知道，不在内存
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    # 预期的总迭代次数。 如果没有意义（None），仅显示基本进度统计信息（无 ETA）。 
                    self.corpus_lines += 1 

            if on_memory:
                self.lines = [line[:-1].split("\\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)
            # 在0-1000行中随便取
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()  # 下一行

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        # 句子1加句子2 中前seq_len个
        segment_label = ([1 for _ in range(len(t1))] +
                         [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        # 随机对每个句子做mask
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                # output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index)) 应该是这个
                output_label.append(0) # todo ？？？

        return tokens, output_label

    # 50%的概率拼接错误句子
    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
