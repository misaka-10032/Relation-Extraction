import pickle
def load_data(file_path):
    """
    Args:
        file_path: path to the data
    Returns:
        data: list of instances in the form of:
                ((token1, token2, ...),
                    (entity1_pos, entity2_pos),
                    relation,
                    reversed)
              reversed = True means relation(e2, e1)
    """
    return pickle.load(open(filepath))

def load_id2vec(dict_path):
    """
    Args:
        dict_path: path to dict
    Returns:
        dictionary of word2vec. key: word id, val: vector
    """
    return pickle.load(open(filepath))

if __name__ == "__main__":
    instances = load_data("../data/instances.bin")
    id2vec = load_id2vec("../data/id2vec.bin")
