import numpy as np
from sincfold.utils import bp2dot, dot2bp
from itertools import product

pairs = {'A': 'U',
         'U': 'A',
         'G': 'C',
         'C': 'G'}

VOCABULARY = ["A", "C", "G", "U"]  # This order matters!
KMERVOCAB = [""]
KMERVOCAB += ["".join(p) for p in product(VOCABULARY, repeat=3)]  # 64 tokens (indexacion a partir de pos 1)

def hallar_complemento(token:int):
    three_mer = KMERVOCAB[token]
    rev_comp = "".join([pairs[nt] for nt in three_mer[::-1]])
    return KMERVOCAB.index(rev_comp)

def random_tokenized_rna(tokenized_structure, stem_token):
    string = ""
    stack = []
    for i in tokenized_structure[::3]:
        if i == ")":
            kmer = stack.pop()
        elif i == "(":
            kmer = KMERVOCAB[stem_token]
            # append the complementary kmer
            stack.append("".join([pairs[nt] for nt in kmer[::-1]]))
        else:  # i == "."
            kmer = KMERVOCAB[np.random.randint(1,65)]
        
        string += kmer
    return string

def seq_from_token_list(token_list):
    # replace * in token_list with random kmer (random int between 1 and 64)
    token_list = [np.random.randint(1, 65) if token == '*' else token for token in token_list]
    rna = "".join(KMERVOCAB[token] for token in token_list)
    return rna

def create_sample_file(filename, tokens, dot):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)

        for i in range(1000):
            id = f"prueba{i}"
            seq = seq_from_token_list(tokens)
            pairs_array = dot2bp(dot)
            print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

# print kmer vocab in file, indexed from 1 to 64
def print_kmer_vocab(filename):
    with open(filename, 'w') as vocab_file:
        for i, kmer in enumerate(KMERVOCAB[1:], start=1):
            print(i, kmer, file=vocab_file)

def create_variable_dataset(filename):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)
        dot = ["*"]
        tok_seq_length = 9
        counter = 0

        for _ in range(100):
            #for i in range(1, tok_seq_length-2):
            for i in range(0, tok_seq_length-6):
                id = f"prueba{counter}"
                counter += 1
                tok_seq = dot*i + [1] + dot*5 + [64] + dot*(tok_seq_length-7-i)
                seq = seq_from_token_list(tok_seq)
                dot_notation = "..."*i + "(((" + "..."*5 + ")))" + "..."*(tok_seq_length-7-i)
                pairs_array = dot2bp(dot_notation)
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)
            
            for i in range(0, tok_seq_length-7):
                id = f"prueba{counter}"
                counter += 1
                tok_seq = dot*i + [1] + dot*6 + [64] + dot*(tok_seq_length-8-i)
                seq = seq_from_token_list(tok_seq)
                dot_notation = "..."*i + "(((" + "..."*6 + ")))" + "..."*(tok_seq_length-8-i)
                pairs_array = dot2bp(dot_notation)
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

            for i in range(0, tok_seq_length-8):
                id = f"prueba{counter}"
                counter += 1
                tok_seq = dot*i + [1] + dot*7 + [64] + dot*(tok_seq_length-9-i)
                seq = seq_from_token_list(tok_seq)
                dot_notation = "..."*i + "(((" + "..."*7 + ")))" + "..."*(tok_seq_length-9-i)
                pairs_array = dot2bp(dot_notation)
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)
        
            # for i in range(1, tok_seq_length-3):
            #     id = f"prueba{counter}"
            #     counter += 1
            #     tok_seq = dot*i + [1] + dot*(tok_seq_length-i-3) + [64]
            #     seq = seq_from_token_list(tok_seq)
            #     dot_notation = "..."*i + "(((" + "..."*(tok_seq_length-i-3) + ")))"
            #     pairs_array = dot2bp(dot_notation)
            #     print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

        for _ in range(100):
            for i in range(0, tok_seq_length-6):
                id = f"prueba{counter}"
                counter += 1
                tok_seq = dot*i + [64] + dot*5 + [1] + dot*(tok_seq_length-7-i)
                seq = seq_from_token_list(tok_seq)
                dot_notation = "..."*i + "(((" + "..."*5 + ")))" + "..."*(tok_seq_length-7-i)
                pairs_array = dot2bp(dot_notation)
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

            for i in range(0, tok_seq_length-7):
                id = f"prueba{counter}"
                counter += 1
                tok_seq = dot*i + [64] + dot*6 + [1] + dot*(tok_seq_length-8-i)
                seq = seq_from_token_list(tok_seq)
                dot_notation = "..."*i + "(((" + "..."*6 + ")))" + "..."*(tok_seq_length-8-i)
                pairs_array = dot2bp(dot_notation)
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

            for i in range(0, tok_seq_length-8):
                id = f"prueba{counter}"
                counter += 1
                tok_seq = dot*i + [64] + dot*7 + [1] + dot*(tok_seq_length-9-i)
                seq = seq_from_token_list(tok_seq)
                dot_notation = "..."*i + "(((" + "..."*7 + ")))" + "..."*(tok_seq_length-9-i)
                pairs_array = dot2bp(dot_notation)
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)
        
def n_token_stem_dataset(filename, symmetric=True):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)
        dot = ["*"]
        tok_seq_length = 12
        stem_size = 5

        counter = 0
        for _ in range(100):
            for z in range(1, (tok_seq_length-2)+1):
                for i in range(0, tok_seq_length-z-2*stem_size+1):
                    id = f"prueba{counter}"
                    counter += 1
                    tok_seq = dot*i + [1]*stem_size + dot*z + [64]*stem_size + dot*(tok_seq_length-z-i-2*stem_size)
                    seq = seq_from_token_list(tok_seq)
                    dot_notation = "..."*i + "((("*stem_size + "..."*z + ")))"*stem_size + "..."*(tok_seq_length-z-i-2*stem_size)
                    pairs_array = dot2bp(dot_notation)
                    print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)
                    
            # include the symmetric case where the stem tokens are swapped
            if symmetric:
                for z in range(1, (tok_seq_length-2)+1):
                    for i in range(0, tok_seq_length-z-2*stem_size+1):
                        id = f"prueba{counter}"
                        counter += 1
                        tok_seq = dot*i + [64]*stem_size + dot*z + [1]*stem_size + dot*(tok_seq_length-z-i-2*stem_size)
                        seq = seq_from_token_list(tok_seq)
                        dot_notation = "..."*i + "((("*stem_size + "..."*z + ")))"*stem_size + "..."*(tok_seq_length-z-i-2*stem_size)
                        pairs_array = dot2bp(dot_notation)
                        print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

def diverse_token_stems(filename):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)
        dot = ["*"]
        tok_seq_length = 12
        stem_size = 1

        counter = 0
        for _ in range(10):
            for three_mer in KMERVOCAB[1:]:
                tok = KMERVOCAB.index(three_mer)
                kot = hallar_complemento(tok)
                for z in range(1, (tok_seq_length-2)+1):
                    for i in range(0, tok_seq_length-z-2*stem_size+1):
                        id = f"prueba{counter}"
                        counter += 1
                        tok_seq = dot*i + [tok]*stem_size + dot*z + [kot]*stem_size + dot*(tok_seq_length-z-i-2*stem_size)
                        seq = seq_from_token_list(tok_seq)
                        dot_notation = "..."*i + "((("*stem_size + "..."*z + ")))"*stem_size + "..."*(tok_seq_length-z-i-2*stem_size)
                        pairs_array = dot2bp(dot_notation)
                        print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)
                        
                # include the symmetric case where the stem tokens are swapped
                #for z in range(1, (tok_seq_length-2)+1):
                    #for i in range(0, tok_seq_length-z-2*stem_size+1):
                # for i in range(4,5):
                #         z = 1

                #         id = f"prueba{counter}"
                #         counter += 1
                #         tok_seq = dot*i + [kot]*stem_size + dot*z + [tok]*stem_size + dot*(tok_seq_length-z-i-2*stem_size)
                #         seq = seq_from_token_list(tok_seq)
                #         dot_notation = "..."*i + "((("*stem_size + "..."*z + ")))"*stem_size + "..."*(tok_seq_length-z-i-2*stem_size)
                #         pairs_array = dot2bp(dot_notation)
                #         print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)



if __name__ == '__main__':
    #dot1 = "......((((((......))))))......"
    #print(len(dot1))
    dot = ['*']
    
    # Tanda 1
    #toq_seq = dot*2 + [1] + [1] + dot*2 + [64] + [64] + dot*2
    #dot_notation = "......((((((......))))))......"

    # Tanda 2
    #toq_seq = dot*2 + [1] + [64] + dot*2 + [1] + [64] + dot*2
    #dot_notation = "......((((((......))))))......"

    # Tanda 3
    #toq_seq = dot*2 + [1] + [43] + dot*2 + [22] + [64] + dot*2
    #dot_notation = "......((((((......))))))......"

    # Tanda 4
    #toq_seq = [1] + [43] + dot*3 + [22] + [64] + dot*4
    #dot_notation = "((((((.........))))))............"
    
    # Tanda 5
    #toq_seq = [1] + dot*6 + [64] + dot*3
    #dot_notation = "(((..................)))........."

    # Tanda 6
    #toq_seq = [1] + [39] + dot + [11] + dot*3 + [24] + [26] + dot + [64]
    #dot_notation = "((((((...(((.........))))))...)))"
    
    # Tanda 7 & 8
    # toq_seq = [1] + dot*2 + [1] + dot*3 + [64] + dot*2 + [64]
    # dot_notation = "(((......(((.........)))......)))"
    # create_sample_file("sample/train_tanda_7a.csv", toq_seq, dot_notation)

    # toq_seq = dot + [64] + [1] + dot*2 + [64] + dot*3 + [1] + dot*2 + [64] + [1]
    # dot_notation = "...((((((......(((.........)))......))))))"
    # create_sample_file("sample/train_tanda_7b.csv", toq_seq, dot_notation)

    # toq_seq = dot + [64] + [64] + [1] + dot*3 + [64] + [1] + [1] + dot*2
    # dot_notation = "...(((((((((.........)))))))))......"
    # create_sample_file("sample/train_tanda_7c.csv", toq_seq, dot_notation)

    # Tanda 9
    #create_variable_dataset("sample/train_tanda_15.csv")
    #n_token_stem_dataset("sample/train_tanda_16.csv")
    diverse_token_stems("sample/train_tanda_17.csv")




    #print_kmer_vocab("sample/kmer_vocab.txt")
