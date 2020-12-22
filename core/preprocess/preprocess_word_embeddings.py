import pickle
import numpy as np


def reformat_glove(glove_file, dim=300):
    print("Loading embeddings...")

    vec_dict = {}

    with open(glove_file, 'r') as f:
        for i, line in enumerate(f):
            if (i + 1) % 10000 == 0:
                print('\tProcessed {}'.format(i + 1))

            splitLine = line.split()
            word = splitLine[0]
            rest = splitLine[-dim::]

            embedding = np.array([float(val) for val in rest])
            vec_dict[word] = embedding

        print("Done. {} embeddings loaded!".format(len(vec_dict)))

    return vec_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', default='data/glove.840B.300d.txt')
    parser.add_argument('--outfile', default='data/glove.840B.300d.pt')

    args = parser.parse_args()

    vecs = reformat_glove(args.infile)

    with open(args.outfile, 'wb') as outf:
        pickle.dump(vecs, outf)
