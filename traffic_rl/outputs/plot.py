# import argparse
# import glob
# from itertools import cycle

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns

# sns.set(
#     style="darkgrid",
#     rc={
#         "figure.figsize": (7.2, 4.45),
#         "text.usetex": False,
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "font.size": 15,
#         "figure.autolayout": True,
#         "axes.titlesize": 16,
#         "axes.labelsize": 17,
#         "lines.linewidth": 2,
#         "lines.markersize": 6,
#         "legend.fontsize": 15,
#     },
# )
# colors = sns.color_palette("colorblind", 4)
# # colors = sns.color_palette("Set1", 2)
# # colors = ['#FF4500','#e31a1c','#329932', 'b', 'b', '#6a3d9a','#fb9a99']
# dashes_styles = cycle(["-", "-.", "--", ":"])
# sns.set_palette(colors)
# colors = cycle(colors)

# def moving_average(interval, window_size):
#     if window_size == 1:
#         return interval
#     window = np.ones(int(window_size)) / float(window_size)
#     return np.convolve(interval, window, "same")

# def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
#     df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")  # convert NaN string to NaN value

#     mean = df.groupby(xaxis).mean()[yaxis]
#     std = df.groupby(xaxis).std()[yaxis]
#     if ma > 1:
#         mean = moving_average(mean, ma)
#         std = moving_average(std, ma)

#     x = df.groupby(xaxis)[xaxis].mean().keys().values
#     plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
#     plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

# if __name__ == "__main__":
#     prs = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
#     )
#     prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
#     prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
#     prs.add_argument("-t", type=str, default="", help="Plot title\n")
#     prs.add_argument("-yaxis", type=str, default="system_total_waiting_time", help="The column to plot.\n")
#     prs.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
#     prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
#     prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
#     prs.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
#     prs.add_argument("-ylabel", type=str, default="Total waiting time (s)", help="Y axis label.\n")
#     prs.add_argument("-output", type=str, default=None, help="PDF output filename.\n")

#     args = prs.parse_args()
#     labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])

#     plt.figure()

#     # File reading and grouping
#     main_df = pd.DataFrame()
#     current_max_timestep = 0
#     for file in args.f:
#         for f in glob.glob(file + "*"):
#             df = pd.read_csv(f, sep=args.sep, error_bad_lines=False)
#             if not main_df.empty:
#                 # Offset the timestep for the new dataframe
#                 df[args.xaxis] += current_max_timestep
#             # Update the current max timestep to reflect the new end of timesteps
#             current_max_timestep = df[args.xaxis].max()
#             main_df = pd.concat((main_df, df))

#     # Plot DataFrame
#     plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)

#     plt.title(args.t)
#     plt.ylabel(args.ylabel)
#     plt.xlabel(args.xlabel)
#     plt.ylim(bottom=0)

#     if args.output is not None:
#         plt.savefig(args.output + ".pdf", bbox_inches="tight")

#     plt.show()
# import argparse
# import glob
# from itertools import cycle
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns

# sns.set(
#     style="darkgrid",
#     rc={
#         "figure.figsize": (7.2, 4.45),
#         "text.usetex": False,
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "font.size": 15,
#         "figure.autolayout": True,
#         "axes.titlesize": 16,
#         "axes.labelsize": 17,
#         "lines.linewidth": 2,
#         "lines.markersize": 6,
#         "legend.fontsize": 15,
#     },
# )
# colors = sns.color_palette("colorblind", 4)
# dashes_styles = cycle(["-", "-.", "--", ":"])
# sns.set_palette(colors)
# colors = cycle(colors)

# def moving_average(interval, window_size):
#     if window_size == 1:
#         return interval
#     window = np.ones(int(window_size)) / float(window_size)
#     return np.convolve(interval, window, "same")

# def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
#     df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")
#     mean = df.groupby(xaxis).mean()[yaxis]
#     std = df.groupby(xaxis).std()[yaxis]
#     if ma > 1:
#         mean = moving_average(mean, ma)
#         std = moving_average(std, ma)
#     x = df.groupby(xaxis)[xaxis].mean().keys().values
#     plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
#     plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

# def print_metrics(df, yaxis, output_file=None, rolling_window=100):
#     # Drop NaNs and filter out zero waiting times for accurate rolling averages
#     df = df[[yaxis]].dropna()
#     df = df[df[yaxis] > 0]  # Exclude zero values

#     # Calculate rolling averages over the specified window
#     rolling_avg = df[yaxis].rolling(window=rolling_window, min_periods=1).mean()
    
#     # Find the max and min of the rolling averages
#     max_avg_wait = rolling_avg.max()  # Highest average waiting time
#     min_avg_wait = rolling_avg.min()  # Lowest average waiting time achieved
    
#     # Calculate the percentage reduction from max to min averages
#     reduction_percentage = ((max_avg_wait - min_avg_wait) / max_avg_wait) * 100 if max_avg_wait != 0 else 0
    
#     # Additional statistics for context
#     overall_mean_wait = df[yaxis].mean()
#     overall_std_wait = df[yaxis].std()
    
#     results = f"""
#     Traffic Signal Control Metrics:
#     --------------------------------
#     Maximum average waiting time: {max_avg_wait:.2f}
#     Minimum average waiting time: {min_avg_wait:.2f}
#     Percentage reduction in average waiting time: {reduction_percentage:.2f}%
#     Overall mean waiting time: {overall_mean_wait:.2f}
#     Overall standard deviation of waiting times: {overall_std_wait:.2f}
#     """
    
#     print(results)
#     if output_file:
#         with open(output_file, "w") as f:
#             f.write(results)

# # Rest of the code remains the same as before


# if __name__ == "__main__":
#     prs = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
#     )
#     prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
#     prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
#     prs.add_argument("-t", type=str, default="", help="Plot title\n")
#     prs.add_argument("-yaxis", type=str, default="system_total_waiting_time", help="The column to plot.\n")
#     prs.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
#     prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
#     prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
#     prs.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
#     prs.add_argument("-ylabel", type=str, default="Total waiting time (s)", help="Y axis label.\n")
#     prs.add_argument("-output", type=str, default=None, help="PDF output filename.\n")
#     prs.add_argument("-metrics_output", type=str, default=None, help="Metrics output filename (optional)\n")

#     args = prs.parse_args()
#     labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])

#     plt.figure()

#     # File reading and grouping
#     main_df = pd.DataFrame()
#     current_max_timestep = 0
#     for file in args.f:
#         for f in glob.glob(file + "*"):
#             df = pd.read_csv(f, sep=args.sep, on_bad_lines="skip")
#             if not main_df.empty:
#                 df[args.xaxis] += current_max_timestep
#             current_max_timestep = df[args.xaxis].max()
#             main_df = pd.concat((main_df, df))

#     # Print metrics
#     print_metrics(main_df, yaxis=args.yaxis, output_file=args.metrics_output)

#     # Plot DataFrame
#     plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)

#     plt.title(args.t)
#     plt.ylabel(args.ylabel)
#     plt.xlabel(args.xlabel)
#     plt.ylim(bottom=0)

#     if args.output is not None:
#         plt.savefig(args.output + ".pdf", bbox_inches="tight")

#     plt.show()
import argparse
import glob
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (7.2, 4.45),
        "text.usetex": False,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 15,
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 15,
    },
)
colors = sns.color_palette("colorblind", 4)
dashes_styles = cycle(["-", "-.", "--", ":"])
sns.set_palette(colors)
colors = cycle(colors)

def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")

def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    # Ensure y-axis data is numeric
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")
    df = df.dropna(subset=[yaxis])  # Drop rows with NaNs

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]

    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values
    plt.plot(x, mean, label=label, color=color, linestyle=next(dashes_styles))
    plt.fill_between(x, mean + std, mean - std, alpha=0.25, color=color, rasterized=True)

def print_metrics(df, yaxis, output_file=None, rolling_window=100):
    # Drop NaNs and filter out zero waiting times for accurate rolling averages
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")
    df = df.dropna(subset=[yaxis])
    df = df[df[yaxis] > 0]  # Exclude zero values

    # Calculate rolling averages over the specified window
    rolling_avg = df[yaxis].rolling(window=rolling_window, min_periods=1).mean()

    # Find the max and min of the rolling averages
    max_avg_wait = rolling_avg.max()  # Highest average waiting time
    min_avg_wait = rolling_avg.min()  # Lowest average waiting time achieved

    # Calculate the percentage reduction from max to min averages
    reduction_percentage = ((max_avg_wait - min_avg_wait) / max_avg_wait) * 100 if max_avg_wait != 0 else 0

    # Additional statistics for context
    overall_mean_wait = df[yaxis].mean()
    overall_std_wait = df[yaxis].std()

    results = f"""
    Traffic Signal Control Metrics:
    --------------------------------
    Maximum average waiting time: {max_avg_wait:.2f}
    Minimum average waiting time: {min_avg_wait:.2f}
    Percentage reduction in average waiting time: {reduction_percentage:.2f}%
    Overall mean waiting time: {overall_mean_wait:.2f}
    Overall standard deviation of waiting times: {overall_std_wait:.2f}
    """

    print(results)
    if output_file:
        with open(output_file, "w") as f:
            f.write(results)

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Plot Traffic Signal Metrics"""
    )
    prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
    prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
    prs.add_argument("-t", type=str, default="", help="Plot title\n")
    prs.add_argument("-yaxis", type=str, default="system_total_waiting_time", help="The column to plot.\n")
    prs.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument("-sep", type=str, default=",", help="Values separator on file.\n")
    prs.add_argument("-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n")
    prs.add_argument("-ylabel", type=str, default="Total waiting time (s)", help="Y axis label.\n")
    prs.add_argument("-output", type=str, default=None, help="PDF output filename.\n")
    prs.add_argument("-metrics_output", type=str, default=None, help="Metrics output filename (optional)\n")

    args = prs.parse_args()
    labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])

    plt.figure()

    # File reading and grouping
    main_df = pd.DataFrame()
    current_max_timestep = 0
    for file in args.f:
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f, sep=args.sep, on_bad_lines="skip")
            # Convert columns to numeric where possible
            df[args.xaxis] = pd.to_numeric(df[args.xaxis], errors="coerce")
            df[args.yaxis] = pd.to_numeric(df[args.yaxis], errors="coerce")
            
            # Check for and drop NaNs
            df = df.dropna(subset=[args.xaxis, args.yaxis])
            
            if not main_df.empty:
                df[args.xaxis] += current_max_timestep
            current_max_timestep = df[args.xaxis].max()
            main_df = pd.concat((main_df, df))

    # Print metrics
    print_metrics(main_df, yaxis=args.yaxis, output_file=args.metrics_output)

    # Plot DataFrame
    plot_df(main_df, xaxis=args.xaxis, yaxis=args.yaxis, label=next(labels), color=next(colors), ma=args.ma)

    plt.title(args.t)
    plt.ylabel(args.ylabel)
    plt.xlabel(args.xlabel)
    plt.ylim(bottom=0)

    if args.output is not None:
        plt.savefig(args.output + ".pdf", bbox_inches="tight")

    plt.show()