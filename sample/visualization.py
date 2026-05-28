from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sincfold.utils import dot2matrix
import seaborn as sns
from sincfold.utils import bp2dot
import os
import argparse

# Forma de usar el script
# $ python sample/visualization 17_test Experimentos/exp_111

# Manage parser
parser = argparse.ArgumentParser(description="Creacion de pdfs con las predicciones del modelo en forma grafica (mapas de calor)")
parser.add_argument("tanda", help="Dataset empleado para el test")
parser.add_argument("exp_dir", help="Directorio con los resultados")
args = parser.parse_args()

#exp_dir = "Experiments/experiment_103/"
#data_dir = exp_dir + "matrices/"


#sample_name = "prueba0"
#sample_name = [f"prueba{i}" for i in range(7)]
#sample_name = [f"prueba{i}" for i in range(60)]
#sample_name = [f"prueba{i}" for i in range(6)] + [f"prueba{i}" for i in range(500,506)]
#sample_name = [f"prueba{i}" for i in range(3)] + [f"prueba{i}" for i in range(300,303)] + [f"prueba{i}" for i in range(600,603)]
#sample_name = [f"prueba{i}" for i in [0,1,2, 65,66,67, 101,102,103, 122,123,124, 132,133,134,
#                                      55,56,57, 180,181,182, 216,217,218, 237,238,239, 247,248,249]]
#sample_name = [f"prueba{i}" for i in range(64)]


def save_heat_map(mat, filename, title):
    plt.figure(figsize=(8, 8))
    sns.heatmap(mat, annot=False, linewidths=.5, linecolor='black', cmap='viridis')
    plt.title(title)
    # remove axis labels
    plt.xticks([])
    plt.yticks([])
    # remove colorbar
    plt.gca().collections[0].colorbar.remove()
    plt.savefig(filename)

def dot_to_kmer_dot(dot):
    kmer_dot = ""
    for i in range(0, len(dot), 3):
        kmer_dot += dot[i]
    return kmer_dot

# TODO: adapt this function
def visualize_single_sample(sample_name, tanda):
    mat_1 = np.loadtxt("middle_matrix/compressed_" + sample_name + ".csv", delimiter=",")
    mat_2 = np.loadtxt("middle_matrix/expanded_" + sample_name + ".csv", delimiter=",")
    mat_3 = np.loadtxt("middle_matrix/prefinal_" + sample_name + ".csv", delimiter=",")

    dot = ""
    with open(f"sample/train_tanda_{tanda}.csv", "r") as file:
        file.readline()
        while line:=file.readline():
            line, pairs, _ = line.strip().split('"')
            name, seq, _ = line.split(",")
            if sample_name in name:
                # convert pairs to list of tuples and to dotbracket notation
                dot = bp2dot(eval(pairs), len(seq))
                break

    ideal = dot2matrix(dot).to('cpu').numpy()
    
    # if direcotry imgs does not exist, create it
    if not os.path.exists("imgs"):
        os.makedirs("imgs")

    save_heat_map(mat_1, "imgs/"+sample_name+"_compressed.png", f"Compressed Matrix - {sample_name}")
    save_heat_map(mat_2, "imgs/"+sample_name+"_expanded.png", f"Expanded Matrix - {sample_name}")
    save_heat_map(mat_3, "imgs/"+sample_name+"_final.png", f"Final Matrix - {sample_name}")
    save_heat_map(ideal, "imgs/"+sample_name+"_real_original.png", f"Real Matrix - {sample_name}")

    # guardar matriz real tokenizada
    ideal_tokenizado = dot2matrix(dot_to_kmer_dot(dot)).to('cpu').numpy()
    save_heat_map(ideal_tokenizado, "imgs/"+sample_name+"_real_tokenized.png", f"Real Tokenized Matrix - {sample_name}")

    # read the csv file from output_trial_3/train_log.csv as dataframe and plot the  

def visualize_many_samples(sample_names, tanda, exp_dir):
    mats_per_page = 6
    cols = 5
    cell_size = 2.5
    label_space = 0.10

    exp_dir = Path(exp_dir)
    data_dir = exp_dir / "matrices/"
    output_pdf = exp_dir/f"test_tanda_{tanda}.pdf"
    
    def load_sample_data(name):
        mat_1 = np.loadtxt(data_dir / f"compressed_{name}.csv", delimiter=",")
        mat_2 = np.loadtxt(data_dir / f"expanded_{name}.csv", delimiter=",")
        mat_3 = np.loadtxt(data_dir / f"final_{name}.csv", delimiter=",")

        # remove bias column if present
        if mat_1.shape[0] < mat_1.shape[1]:
            mat_1 = mat_1[:,:mat_1.shape[0]]
        
        dot = ""
        with open(f"sample/train_tanda_{tanda}.csv", "r") as file:
            file.readline()
            while line := file.readline():
                line, pairs, _ = line.strip().split('"')
                name_col, seq, _ = line.split(",")
                if name in name_col:
                    dot = bp2dot(eval(pairs), len(seq))
                    break
        
        ideal = dot2matrix(dot).to('cpu').numpy()
        ideal_tokenizado = dot2matrix(dot_to_kmer_dot(dot)).to('cpu').numpy()
        
        return ideal, ideal_tokenizado, mat_1, mat_2, mat_3
    
    all_sample_data = []
    for sample_name in sample_names:
        try:
            data = load_sample_data(sample_name)
            all_sample_data.append((sample_name, data))
        except Exception as e:
            print(f"Error loading sample {sample_name}: {e}")
    
    titles = ["Ground Truth", "Ground Truth Tokenizado", "Matriz de Atención", "Unpooled", "Final"]
    
    with PdfPages(output_pdf) as pdf:
        for page_start in range(0, len(all_sample_data), mats_per_page):
            page_data = all_sample_data[page_start:page_start + mats_per_page]
            
            fig_height = mats_per_page * cell_size + 1.0
            fig, axes = plt.subplots(mats_per_page, cols, figsize=(11.69, fig_height))
            fig.suptitle("RNA Secondary Structure Prediction (Training by overfitting)", fontsize=12, y=0.99)
            
            for row_idx, (sample_name, (mat_real, mat_real_tokenized, mat_compressed, mat_expanded, mat_final)) in enumerate(page_data):
                mats = [mat_real, mat_real_tokenized, mat_compressed, mat_expanded, mat_final]
                
                for col_idx, (mat, title) in enumerate(zip(mats, titles)):
                    ax = axes[row_idx, col_idx]

                    if col_idx == 1:
                        step = 1
                        xticks = list(range(0, mat.shape[1], step))
                        yticks = list(range(0, mat.shape[0], step))
                        sns.heatmap(mat, annot=False, linewidths=0.1, linecolor='black', cmap='plasma', ax=ax,
                               xticklabels=xticks, yticklabels=yticks, square=True)
                        ax.tick_params(left=True, bottom=True, labelsize=6)
                    elif col_idx == 2:
                        step = 1
                        xticks = list(range(0, mat.shape[1], step))
                        yticks = list(range(0, mat.shape[0], step))
                        sns.heatmap(mat, annot=False, linewidths=0.1, linecolor='black', cmap='plasma', ax=ax,
                               xticklabels=xticks, yticklabels=yticks, square=True)
                        ax.tick_params(left=True, bottom=True)
                    else:
                        sns.heatmap(mat, annot=False, linewidths=0.1, linecolor='black', cmap='plasma', ax=ax,
                               xticklabels=False, yticklabels=False, square=True)
                        ax.tick_params(left=False, bottom=False)

                    if row_idx == 0:
                        ax.set_title(title, fontsize=8)
                    ax.tick_params(left=False, bottom=False)
                    
                    if col_idx == 0:
                        ax.set_ylabel(sample_name, fontsize=7, rotation=0, labelpad=30, va='center')
                    
                    ax.collections[0].colorbar.remove()
            
            for empty_row in range(len(page_data), mats_per_page):
                for col_idx in range(cols):
                    axes[empty_row, col_idx].axis('off')
            
            plt.subplots_adjust(hspace=0.02, wspace=0.15, left=label_space, right=0.99, top=0.95, bottom=0.02)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            
            print(f"Added page with {len(page_data)} samples")
    
    print(f"PDF saved to {output_pdf}")

# TODO: adapt this function
def visualize_heads_per_sample(sample_names, tanda, output_pdf="sample/heads_visualization.pdf"):
    """
    Generate pdf with one page per 4 samples, each page with 8 columns:
    * Ground Truth
    * Ground Truth Tokenizado
    * Matriz de Atención Head 1
    * Matriz de Atención Head 2
    * Matriz de Atención Head 3
    * Matriz de Atención Head 4
    * Unpooled
    * Simetrico
    """
    mats_per_page = 4
    cols = 8
    cell_size = 1.8
    label_space = 0.08
    
    def load_sample_data(name):
        #mat_1 = np.loadtxt(f"middle_matrix/compressed_{name}.csv", delimiter=",")
        mat_2 = np.loadtxt(f"middle_matrix/expanded_{name}.csv", delimiter=",")
        mat_3 = np.loadtxt(f"middle_matrix/prefinal_{name}.csv", delimiter=",")
        
        head_mats = []
        for h in range(4):
            try:
                head_mat = np.loadtxt(f"middle_matrix/compressed_{name}_head_{h}.csv", delimiter=",")
                head_mats.append(head_mat)
            except Exception as e:
                print(f"Error loading head {h} for sample {name}: {e}")
        
        dot = ""
        with open(f"sample/train_tanda_{tanda}.csv", "r") as file:
            file.readline()
            while line := file.readline():
                line, pairs, _ = line.strip().split('"')
                name_col, seq, _ = line.split(",")
                if name in name_col:
                    dot = bp2dot(eval(pairs), len(seq))
                    break
        
        ideal = dot2matrix(dot).to('cpu').numpy()
        ideal_tokenizado = dot2matrix(dot_to_kmer_dot(dot)).to('cpu').numpy()
        
        return ideal, ideal_tokenizado, head_mats[0], head_mats[1], head_mats[2], head_mats[3], mat_2, mat_3
    
    all_sample_data = []
    for sample_name in sample_names:
        try:
            data = load_sample_data(sample_name)
            all_sample_data.append((sample_name, data))
        except Exception as e:
            print(f"Error loading sample {sample_name}: {e}")
    
    titles = ["Ground Truth", "Ground Truth Tokenizado", "Head 1", "Head 2", "Head 3", "Head 4", "Unpooled", "Symmetric"]
    
    with PdfPages(output_pdf) as pdf:
        for page_start in range(0, len(all_sample_data), mats_per_page):
            page_data = all_sample_data[page_start:page_start + mats_per_page]
            
            fig_height = mats_per_page * cell_size + 0.4
            fig, axes = plt.subplots(mats_per_page, cols, figsize=(11.69, fig_height))
            fig.suptitle("RNA Secondary Structure - Attention Heads Analysis", fontsize=12, y=0.99)
            
            for row_idx, (sample_name, data) in enumerate(page_data):
                (mat_real, mat_real_tokenized, head1, head2, head3, head4, mat_compressed, mat_final) = data
                mats = [mat_real, mat_real_tokenized, head1, head2, head3, head4, mat_compressed, mat_final]
                
                for col_idx, (mat, title) in enumerate(zip(mats, titles)):
                    ax = axes[row_idx, col_idx]
                    sns.heatmap(mat, annot=False, linewidths=0, cmap='plasma', ax=ax, 
                               xticklabels=False, yticklabels=False, square=True)
                    if row_idx == 0:
                        ax.set_title(title, fontsize=8)
                    ax.tick_params(left=False, bottom=False)
                    
                    if col_idx == 0:
                        ax.set_ylabel(sample_name, fontsize=7, rotation=0, labelpad=25, va='center')
                    
                    ax.collections[0].colorbar.remove()
            
            for empty_row in range(len(page_data), mats_per_page):
                for col_idx in range(cols):
                    axes[empty_row, col_idx].axis('off')
            
            plt.subplots_adjust(hspace=0.02, wspace=0.01, left=label_space, right=0.99, top=0.95, bottom=0.02)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            
            print(f"Added page with {len(page_data)} samples")
    
    print(f"PDF saved to {output_pdf}")

def get_train_dataset(output_dir):
    with open(output_dir / "exp_config.txt", "r") as file:
        for line in file:
            if "train_file" in line:
                train_file = line.split(":")[1].strip()
                break
    tanda = train_file[train_file.rfind("\\")+13:-4]
    return tanda

def choose_samples(tanda):
    sample_name = []

    if tanda in ["1", "2", "3", "4", "5", "6", "8"]:
        sample_name = [f"prueba{i}" for i in range(7)]
    elif tanda in ["7_mix1", "7_mix2"]:
        sample_name = [f"prueba{i}" for i in range(3)] + [f"prueba{i}" for i in range(300,303)] + [f"prueba{i}" for i in range(600,603)]
    elif tanda in ["7_mix3", "7_mix4", "10"]:
        sample_name = [f"prueba{i}" for i in range(6)] + [f"prueba{i}" for i in range(500,506)]
    elif tanda in ["9"]:
        sample_name = [f"prueba{i}" for i in range(13)]
    elif tanda in ["11a", "11b", "12a", "12b", "13a", "13b", "14a", "14b", "14c", "14d", "14e"]:
        sample_name = [f"prueba{i}" for i in range(7)]
    elif tanda in ["14_testeo"]:
        sample_name = [f"prueba{i}" for i in range(28)]
    elif tanda in ["15a"]:
        sample_name = [f"prueba{i}" for i in range(7)] + [f"prueba{i}" for i in range(700, 707)]
    elif tanda in ["15b"]:
        sample_name = [f"prueba{i}" for i in range(6)] + [f"prueba{i}" for i in range(600, 606)]
    elif tanda in ["15c"]:
        sample_name = [f"prueba{i}" for i in range(5)] + [f"prueba{i}" for i in range(500, 505)]
    elif tanda in ["15d"]:
        sample_name = [f"prueba{i}" for i in range(4)] + [f"prueba{i}" for i in range(400, 404)]
    elif tanda in ["15e"]:
        sample_name = [f"prueba{i}" for i in range(6)] + [f"prueba{i}" for i in range(600, 606)]
    elif tanda in ["16a"]:
        sample_name = [f"prueba{i}" for i in range(110)]
    elif tanda in ["16b"]:
        sample_name = [f"prueba{i}" for i in range(72)]
    elif tanda in ["16c"]:
        sample_name = [f"prueba{i}" for i in range(42)]
    elif tanda in ["16d"]:
        sample_name = [f"prueba{i}" for i in range(20)]
    elif tanda in ["16e"]:
        sample_name = [f"prueba{i}" for i in range(6)]
    elif tanda in ["16_testeo"]:
        sample_name = [f"prueba{i}" for i in range(250)]
    elif tanda in ["17_testeo"]:
        sample_name = [f"prueba{i}" for i in range(64)]
    elif "17_" in tanda:
        num = int(tanda.split("_")[1])
        sample_name = []
        counter = 0
        for _ in range(num):
            for i in range(10,0,-1):
                sample_name.append(f"prueba{counter}")
                counter += i
    elif tanda == 'variable_stems':
        sample_name = [f"prueba{i}" for i in range(40)]
    else:
        sample_name = [f"prueba{i}" for i in range(7)]

    return sample_name


if __name__ == "__main__":
    ## visualize_single_sample(sample_name, tanda)
    # visualize_many_samples(sample_name, tanda, output_pdf=exp_dir+"test_visualization.pdf")
    ## visualize_heads_per_sample(sample_name, tanda)

    tanda = args.tanda
    exp_dir = Path(args.exp_dir)

    print(f"Output dir: {exp_dir}")
    print(f"Tanda: {tanda}")

    samples = choose_samples(tanda)
    print(f"Amount of samples: {len(samples)}")
        
    visualize_many_samples(samples, tanda, exp_dir=exp_dir)

    