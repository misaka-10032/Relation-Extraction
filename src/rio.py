import pickle
def load_data(file_path):
    """
    Args:
        file_path: path to the data
    Returns:
        data: list of instances in the form of:
                ((token1_id, token2_id, ...),
                    (entity1_pos, entity2_pos),
                    relation_id)
    """
    return pickle.load(open(file_path))

def load_id2vec(file_path):
    """
    Args:
        dict_path: path to dict
    Returns:
        dictionary of word2vec. key: word id, val: vector
    """
    return pickle.load(open(file_path))

if __name__ == "__main__":
    instances = load_data("../data/instances.bin")
    id2vec = load_id2vec("../data/id2vec.bin")
    label_set = set()
    for entry in instances:
        label_set.add(entry[2])
    print label_set
    print len(id2vec)
