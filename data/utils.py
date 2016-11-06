import re
import pickle
import numpy as np
import string

def relation_to_id(label, is_reversed):
    labels = ['Instrument-Agency',
              'Entity-Origin',
              'Component-Whole',
              'Cause-Effect',
              'Entity-Destination',
              'Product-Producer',
              'Member-Collection',
              'Content-Container',
              'Message-Topic',
              "Other"]
    return labels.index(label) * 2 + is_reversed


def seperate_words_and_symbols(token):
    i = 0
    tokens = list()
    t_token = ""
    while i < len(token):
        if token[i] in string.letters:
            t_token += token[i]
        else:
            if t_token:
                tokens.append(t_token)
            if token[i] != " ":
                tokens.append(token[i])
            t_token = ""
        i += 1
    if t_token:
        tokens.append(t_token)
    return tokens

def get_token_list(sentence):
    e1_start_pos, e2_start_pos = -1, -1
    e1_end_pos, e2_end_pos = -1, -1
    t_list = list()
    cur_index = 0
    for token in sentence:
        token = token.lower()
        tokens = list()

        if "<e1>" in token and "</e1>" in token:
            t1, t2 = token.split("<e1>")
            t2, t3 = t2.split("</e1>")
            tokens1 = seperate_words_and_symbols(t1)
            tokens2 = seperate_words_and_symbols(t2)
            tokens3 = seperate_words_and_symbols(t3)
            e1_start_pos = cur_index + len(tokens1)
            e1_end_pos = e1_start_pos
            tokens = tokens1 + tokens2 + tokens3
        elif "<e2>" in token and "</e2>" in token:
            t1, t2 = token.split("<e2>")
            t2, t3 = t2.split("</e2>")
            tokens1 = seperate_words_and_symbols(t1)
            tokens2 = seperate_words_and_symbols(t2)
            tokens3 = seperate_words_and_symbols(t3)
            e2_start_pos = cur_index + len(tokens1)
            e2_end_pos = e2_start_pos
            tokens = tokens1 + tokens2 + tokens3
        elif "<e1>" in token or "<e2>" in token:
            t1, t2 = "", ""
            if "<e1>" in token:
                t1, t2 = token.split("<e1>")
                tokens1 = seperate_words_and_symbols(t1)
                tokens2 = seperate_words_and_symbols(t2)
                e1_start_pos = cur_index + len(tokens1)
            else:
                t1, t2 = token.split("<e2>")
                tokens1 = seperate_words_and_symbols(t1)
                tokens2 = seperate_words_and_symbols(t2)
                e2_start_pos = cur_index + len(tokens2)
            tokens = tokens1 + tokens2
        elif "</e1>" in token or "</e2>" in token:
            if "</e1>" in token:
                t1, t2 = token.split("</e1>")
                tokens1 = seperate_words_and_symbols(t1)
                tokens2 = seperate_words_and_symbols(t2)
                e1_end_pos = cur_index + len(tokens1) - 1
            else:
                t1, t2 = token.split("</e2>")
                tokens1 = seperate_words_and_symbols(t1)
                tokens2 = seperate_words_and_symbols(t2)
                e2_end_pos = cur_index + len(tokens1) - 1
            tokens = tokens1 + tokens2
        else:
            tokens = seperate_words_and_symbols(token)
        t_list.extend(tokens)
        cur_index = len(t_list)
    result = t_list[:e1_start_pos] \
                + ["_".join(t_list[e1_start_pos:e1_end_pos+1])] \
                + t_list[e1_end_pos+1:e2_start_pos] \
                + ["_".join(t_list[e2_start_pos:e2_end_pos+1])] \
                + t_list[e2_end_pos+1:]
    return result, (e1_start_pos, e2_start_pos - e1_end_pos + e1_start_pos)

def get_vocab_from_examples(data):
    '''
    Args:
        data: list of examples in the form of:
                ((token1, token2, ...),
                    (entity1_pos, entity2_pos),
                    relation_id)
    Returns:
        vocab: a string set of words appearing in data
    '''
    vocab = set()
    for entry in data:
        vocab.update(entry[0])
    return vocab


def parse_data(file_list):
    '''
    Args:
        file_list: A list of filename strings
    Returns:
        data: list of examples in the form of:
                ((token1, token2, ...),
                    (entity1_pos, entity2_pos),
                    relation_id)
    '''
    data = list()
    for file_name in file_list:
        with open(file_name) as f:
            while f.read(1):
                f.seek(-1, 1) # 1 represents cur_pos in file

                #remove number, quote
                sentence = f.readline().strip().split('\t')[1][1:-1]
                sentence = re.sub("(?P<w1>\w+)-(?P<w2>\w+)", "\g<w1> \g<w2>", sentence)
                sentence = ["^"] + sentence.split() + ["#"]
                relation = f.readline().strip()
                comment = f.readline()
                f.readline()    # for the blank line

                token_list, (e1_pos, e2_pos) = get_token_list(sentence)

                e1_relation_pos = relation.find("e1")
                e2_relation_pos = relation.find("e2")

                relation = re.sub("\(e\d.*e\d\)", "", relation)
                is_reversed = e1_relation_pos > e2_relation_pos
                relation_id = relation_to_id(relation, is_reversed)

                # print token_list, token_list[e1_pos], token_list[e2_pos]
                data.append((tuple(token_list),
                             (e1_pos, e2_pos),
                             relation_id))
    return data


def generate_word_to_id_dict(vocab):
    '''
    Args:
        vocab: a set of words appearing in both training and testing data
    Returns:
        result: a word to id dict
    '''
    result = dict()
    for i, word in enumerate(vocab):
        result[word] = i + 1
    return result

def generate_id_to_vec_dict(word2id, word2vec):
    '''
    Args:
        word2id: a word to id dict
        word2vec: a word to vec dict
    Returns:
        result: a id to vec dict
    '''
    WORD_EMBEDDING_LEN = 50
    result = dict()
    for word in word2id:
        if word in word2vec:
            result[word2id[word]] = word2vec[word]
        else:
            result[word2id[word]] = np.random.rand(WORD_EMBEDDING_LEN)
    return result


def get_word_to_vec_dict(wordvec_file_path,
             vocab_set):
    word_to_vec_dict = dict()
    with open(wordvec_file_path) as f:
        line = f.readline()
        for line in f:
            first_space = line.find(" ")
            word, vec = line[:first_space], \
                        np.fromstring(line[first_space+1:], sep=" ")

            if word in vocab_set:
                word_to_vec_dict[word] = vec
    return word_to_vec_dict


def get_id_instances(text_instances, word2id):
    '''
    Args:
        text_instances: list of examples in the form of:
                ((token1, token2, ...),
                    (entity1_pos, entity2_pos),
                    relation_id)
        word2id: a word to id dict

    Returns:
        result: list of examples in the form of:
                ((token1_id, token2_id, ...),
                    (entity1_pos, entity2_pos),
                    relation_id)
    '''
    result = list()
    for entry in text_instances:
        sentence = list()
        for token in entry[0]:
            sentence.append(word2id[token])
        result.append((tuple(sentence), entry[1], entry[2]))
    return result

if __name__ == "__main__":
    text_instances = parse_data(["train.txt", "test.txt"])
    vocab = get_vocab_from_examples(text_instances)
    vocab.add("^") # beginning sign
    vocab.add("$") # ending sign
    word2id = generate_word_to_id_dict(vocab)
    word2vec = get_word_to_vec_dict("vec.txt", vocab)
    id2vec = generate_id_to_vec_dict(word2id, word2vec)
    id_instances = get_id_instances(text_instances, word2id)
    pickle.dump(id_instances, open("instances.bin", "w"))
    pickle.dump(id2vec, open("id2vec.bin", "w"))
    pickle.dump(word2id, open("word2id.bin", "w"))

    print "Label ID:\n"
    labels = ['Instrument-Agency', 'Entity-Origin', 'Component-Whole', 'Cause-Effect', 'Entity-Destination', 'Product-Producer', 'Member-Collection', 'Content-Container', 'Message-Topic', "Other"]
    for label in labels:
        reversed_label = "-".join(reversed(label.split("-")))
        print "%s:%d\t%s:%d" % (label, relation_to_id(label, False), reversed_label, relation_to_id(label, True))

    print "\nInstance Example:\n", id_instances[0]

    print "\n\nID to Vec Example:"
    count = 0
    for k in id2vec:
        print k, id2vec[k]
        if count < 1:
            count += 1
        else: break
    count = 0

    print "\n\nWord to ID Example:"
    for k in word2id:
        print k, word2id[k]
        if count < 1:
            count += 1
        else: break

#print parse_data(["dev.txt"])
# print get_vocab_from_examples(parse_data(["train.txt", "test.txt"]))
