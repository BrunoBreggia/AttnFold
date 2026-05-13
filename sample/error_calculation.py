import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sincfold.utils import bp2matrix
import seaborn as sns
from matplotlib.colors import LogNorm

exp = 80
mat_groud_truth_file = 'sample/train_tanda_14_testeo.csv'
sample = 'prueba0'
mat_pred_file = f'Architecture7/output_trial_{exp}/middle_matrix/compressed_{sample}.csv'


def get_gt_matrix(mat_groud_truth_file, sample, half=False):
    """
    Builds the groud truth matrix
    """
    df = pd.read_csv(mat_groud_truth_file)
    df_sample = df[df['id'] == sample]
    length = len(df_sample['sequence'].values[0])
    base_pairs = df_sample['base_pairs']
    # Convert list in string format to list object
    base_pairs = eval(base_pairs.values[0])
    mat_gt = bp2matrix(length, base_pairs)

    # Get 3-mer tokenized version of matrix
    mat_gt_3mer = np.zeros((length//3, length//3), dtype=int)
    for i in range(length//3):
        for j in range(length//3):
            if mat_gt[i*3+1, j*3+1] != 0:
                mat_gt_3mer[i, j] = 1 

    # if half, zero out upper triangular half of the matrix
    if half:
        mat_gt_3mer = np.tril(mat_gt_3mer)
    return mat_gt_3mer

def get_pred_matrix(mat_pred_file, select_top=0):
    """
    Obains the predicted matrix from the csv file, scales it from 0 to 1 and returns it as a numpy array
    """
    mat_pred = pd.read_csv(mat_pred_file, header=None).values
    # if rectangular matrix, remove last column (k-bias)
    if mat_pred.shape[0] != mat_pred.shape[1]:
        mat_pred = mat_pred[:, :-1]
    # Scale from 0 to 1
    # mat_pred -= np.min(mat_pred)
    # mat_pred = mat_pred / np.max(mat_pred)

    # if select_top != 0: find the top 'select_top' elements in mat_pred and set them to 1, the rest to 0
    if select_top != 0:
        flat = mat_pred.flatten()
        indices = np.argpartition(flat, -select_top)[-select_top:]
        mat_pred = np.zeros_like(mat_pred)
        for index in indices:
            i, j = np.unravel_index(index, mat_pred.shape)
            mat_pred[i, j] = 1
    return mat_pred

def calculate_error(mat_gt, mat_pred):
    """
    Calculates the error between the ground truth matrix and the predicted matrix, as the sum of absolute differences
    """
    error = np.sum(np.abs(mat_gt - mat_pred))
    return error

def graph_matrix(mat, add_label=False, save=False):
    """
    Generates a heatmap of the matrix using seaborn, with a colorbar and gridlines, and saves the figure as a png file
    """
    # use sns.heatmap to graph the matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    step = 1
    xticks = list(range(0, mat.shape[1], step))
    yticks = list(range(0, mat.shape[0], step))
    if not add_label:
        sns.heatmap(mat, annot=False, linewidths=0.1, linecolor='black', cmap='plasma', ax=ax,
                xticklabels=xticks, yticklabels=yticks, square=True)
        ax.tick_params(left=True, bottom=True, labelsize=6)
        ax.collections[0].colorbar.remove() # remove color bar
    else:
        sns.heatmap(mat, annot=True, fmt=".2f", linewidths=0.1, linecolor='black', cmap='plasma', ax=ax,
                xticklabels=xticks, yticklabels=yticks, square=True)
        ax.tick_params(left=True, bottom=True, labelsize=6)
    if save:
        plt.savefig("sample/comparisson_matrix.png")
    else:
        plt.show()

def graph_error(mat_gt, mat_pred, graph_log=False):
    """
    Graphical representation of the error between the groud truth matrix and the predicted matrix, as a heatmap of the absolute differences, with a colorbar and gridlines, and saves the figure as a png file
    If graph_log is True, the color scale is logarithmic, to better visualize the differences when the error values are very small
    """
    error_matrix = np.abs(mat_gt - mat_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    step = 1
    xticks = list(range(0, error_matrix.shape[1], step))
    yticks = list(range(0, error_matrix.shape[0], step))
    if not graph_log:
        sns.heatmap(error_matrix, annot=False, linewidths=0.1, linecolor='black', cmap='Reds', ax=ax,
            xticklabels=xticks, yticklabels=yticks, square=True)
    else:
        sns.heatmap(error_matrix, annot=False, linewidths=0.1, linecolor='black', cmap='Reds', ax=ax, norm=LogNorm(),
                xticklabels=xticks, yticklabels=yticks, square=True)
    ax.tick_params(left=True, bottom=True, labelsize=6)
    #ax.collections[0].colorbar.remove() # remove color bar
    plt.show()

def comparisson_matrix(groud_truths, sample_labels, exps):
    """
    Generates a comparisson matrix between the groud truth matrices and the predicted matrices 
    for different experiments, as a heatmap of the absolute differences.

    """
    zero_matrix = np.zeros((len(sample_labels),)*2, dtype=float)
    comparisson_mat = zero_matrix.copy()

    for i, exp in enumerate(exps):
        counter = 0 # ground truth index
        for j, sublabels in enumerate(sample_labels):

            # Calculate the mean prediction error for a particular stem type
            mean_error_mat = 0
            for k, sublabel in enumerate(sublabels):
                #graph_error(groud_truths[counter], get_pred_matrix(f'Architecture7/output_trial_{exp}/middle_matrix/compressed_{sublabel}.csv', select_top=1), graph_log=False)
                pred_mat = get_pred_matrix(f'../myExperiments/output_trial_{exp}/middle_matrix_100epochs/compressed_{sublabel}.csv', select_top=0)
                mean_error_mat += calculate_error(groud_truths[counter], pred_mat)
                counter += 1
            mean_error_mat /= len(sublabels)

            comparisson_mat[i, j] = mean_error_mat
        
    return comparisson_mat

if __name__ == "__main__":
    #mat_gt = get_gt_matrix(mat_groud_truth_file, sample)
    #graph_matrix(mat_gt)
    #mat_pred = get_pred_matrix(mat_pred_file)
    #graph_matrix(mat_pred)
    #error = calculate_error(mat_gt, mat_pred)
    #print(f'Error: {error}')
    #graph_error(mat_gt, mat_pred, True)

    # Example of comparisson matrix
    ground_truths = [get_gt_matrix(mat_groud_truth_file, sample, half=True) for sample in [f'prueba{i}' for i in range(0, 29)]]
    exps = list(range(95, 100))
    sample_labels = [
        # [f'prueba{i}' for i in range(0, 7)],
        # [f'prueba{i}' for i in range(7, 13)],
        # [f'prueba{i}' for i in range(13, 18)],
        # [f'prueba{i}' for i in range(18, 22)],
        # [f'prueba{i}' for i in range(22, 28)],

        [f'prueba{i}' for i in range(28, 35)],
        [f'prueba{i}' for i in range(35, 41)],
        [f'prueba{i}' for i in range(41, 46)],
        [f'prueba{i}' for i in range(46, 50)],
        [f'prueba{i}' for i in range(50, 56)],
        ]
    comparisson_mat = comparisson_matrix(ground_truths, sample_labels, exps)
    graph_matrix(comparisson_mat, add_label=True, save=True)


    

