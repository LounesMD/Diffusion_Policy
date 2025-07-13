import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_curve_plot_data(data_dict):
    """
    Takes as an input a dictionary {experience_name_id : [(id,step,reward,done)]}
    and returns a list: [(step,reward)]
    """
    result = []
    for _, tuples_list in data_dict.items():
        to_add = []
        to_add.extend((elt[1], elt[2]) for elt in tuples_list)
        if len(tuples_list) < 300:
            last_element = tuples_list[-1][2]
            to_add.extend((i, last_element) for i in range(len(tuples_list) + 1, 301))
        result.append(to_add)
    return result


def make_box_plot_data(data_dict):
    """
    Takes as an input a dictionary {experience_name_id : [(id,step,reward,done)]}
    and returns a list: [nb_steps]
    """
    result = []
    for _, tuples_list in data_dict.items():
        result.append(tuples_list[-1][1])
    return result


def plot_reward_statistics(
    res_plot, q=10, colors=None, label="Rewards", show=True, color="blue"
):
    """
    Plots the mean reward and the shaded region representing percentiles.

    Args:
        res_plot (list of lists): A list of lists where each sublist contains tuples (i, reward).
        q (int): The percentile for the shaded region (default is 10, representing 10th and 90th percentiles).
        colors (str or list): Color for the plot line and shaded region (default is None, which uses Matplotlib defaults).
        label (str): Label for the plot line (default is "Rewards").
    """
    # Flatten res_plot into a dictionary grouped by index i
    reward_data = {}
    for sublist in res_plot:
        for i, reward in sublist:
            reward_data.setdefault(i, []).append(reward)

    # Create arrays for plotting
    indices = sorted(reward_data.keys())
    cumdata = np.array(
        [reward_data[i] for i in indices]
    )  # Rows are rewards for each index i

    # Compute statistics
    mean_reg = np.mean(cumdata, axis=1)  # Mean reward per index
    q_reg = np.percentile(cumdata, q, axis=1)  # Lower percentile
    Q_reg = np.percentile(cumdata, 100 - q, axis=1)  # Upper percentile

    # Plot mean and shaded percentiles
    plt.plot(indices, mean_reg, label=label, color=color)
    plt.fill_between(
        indices,
        q_reg,
        Q_reg,
        color=color,
        alpha=0.15,
        # label=f"{100 - 2*q}th Percentile Range",
    )

    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Reward per time step")
    plt.legend()
    if show:
        plt.grid()
        plt.show()


def plot_boxplot(
    data,
    title="Box Plot",
    xlabel="Nb predicted actions",
    ylabel="Nb steps",
    color="blue",
):
    """
    Creates a box plot from a list of integers and highlights the average with a darker marker.

    Args:
        data (list of int): List of integers to visualize in a box plot.
        title (str): Title of the plot (default: "Box Plot").
        xlabel (str): Label for the x-axis (default: "Nb predicted actions").
        ylabel (str): Label for the y-axis (default: "Nb steps").
        color (str): Color of the box plot (default: "blue").
    """
    # Create the box plot
    plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color, color="black"))

    # Highlight the mean as a darker marker
    mean = np.mean(data)
    plt.scatter(1, mean, color="darkred", zorder=3, label="Mean")  # Add mean marker

    # Customize plot appearance
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


def plot_all():
    # Load the CSV file into a DataFrame
    csv_file_path_0 = [
        "./outputs/results_actionssteps_1_patchsize_0.csv",
        "./outputs/results_actionssteps_2_patchsize_0.csv",
        "./outputs/results_actionssteps_5_patchsize_0.csv",
        "./outputs/results_actionssteps_8_patchsize_0.csv",
        "./outputs/results_actionssteps_15_patchsize_0.csv",
        "./outputs/results_actionssteps_30_patchsize_0.csv",
        "./outputs/results_actionssteps_150_patchsize_0.csv",
    ]
    csv_file_path_8 = [
        "./outputs/results_actionssteps_1_patchsize_8.csv",
        "./outputs/results_actionssteps_2_patchsize_8.csv",
        "./outputs/results_actionssteps_5_patchsize_8.csv",
        "./outputs/results_actionssteps_8_patchsize_8.csv",
        "./outputs/results_actionssteps_15_patchsize_8.csv",
        "./outputs/results_actionssteps_30_patchsize_8.csv",
        "./outputs/results_actionssteps_150_patchsize_8.csv",
    ]

    c = ["blue", "red", "green", "black", "orange", "magenta", "yellow"]
    nb = ["1", "2", "5", "8", "15", "30", "150"]
    for idx, file in enumerate(csv_file_path_8):
        df = pd.read_csv(file)

        # Convert to a dictionary where the key is the column header, and the value is a list of tuples
        results_dict = {
            col: df[col].dropna().apply(eval).tolist() for col in df.columns
        }

        res = make_curve_plot_data(results_dict)
        plot_reward_statistics(
            res, show=False, color=c[idx], label="seq. length: " + nb[idx]
        )
    plt.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize="small")
    plt.grid()
    plt.show()


def main():
    # plot_all()
    parser = argparse.ArgumentParser(
        description="""A script to make plots with data stored in a csv file.""",
    )

    parser.add_argument(
        "--exp_file",
        type=str,
        default="outputs/results.csv",
        help="""File where the results are saved. They will be saved.""",
    )
    args = parser.parse_args()

    # Load the CSV file into a DataFrame
    csv_file_path = args.exp_file
    df = pd.read_csv(csv_file_path)

    # Convert to a dictionary where the key is the column header, and the value is a list of tuples
    results_dict = {col: df[col].dropna().apply(eval).tolist() for col in df.columns}
    res_plot = make_curve_plot_data(results_dict)
    plot_reward_statistics(res_plot)

    res_box = make_box_plot_data(results_dict)
    plot_boxplot(res_box)


if __name__ == "__main__":
    main()
