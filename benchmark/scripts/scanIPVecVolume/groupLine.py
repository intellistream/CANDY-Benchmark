
from matplotlib.lines import Line2D
from matplotlib import font_manager
grouping_rule = {'Baseline':'Baseline:' , 
                 'Flann': 'Tree-based:',
                 'SPTAG': 'Tree-based:',
                 'LSH': 'LSH-based:',
                 'LSHAPG': 'LSH-based:',
                 'PQ': 'Clustering-based:', 
                 'IVFPQ': 'Clustering-based:',
                'onlinePQ':'Clustering-based:', 
                'HNSW': 'Graph-based:', 
                'NSW': 'Graph-based:', 
                'NSG': 'Graph-based:',
                'DPG': 'Graph-based:',
                'freshDiskAnn':'Graph-based:',
                'MNRU':'Graph-based:'}
def DrawFigureYnormal(xvalues, yvalues, legend_labels, x_label, y_label, y_min, y_max, filename, allow_legend):
      # Create a figure with a specified size
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Define line colors for each series
    LINE_COLORS = [
        '#FF8C00', '#FFE4C4', '#00FFFF', '#E0FFFF',
        '#FF6347', '#98FB98', '#800080', '#FFD700',
        '#7CFC00', '#8A2BE2', '#FF4500', '#20B2AA',
        '#B0E0E6', '#00000F', '#00FF7F'
    ]

    # Ensure inputs have matching lengths
    assert len(legend_labels) == len(xvalues) == len(yvalues), "Mismatched lengths of inputs"

    # Get unique groups from the grouping rule in the order they appear
    unique_groups = []
    seen = set()
    for label in legend_labels:
        group = grouping_rule.get(label)
        if group not in seen:
            seen.add(group)
            unique_groups.append(group)
    
    # Prepare legend entries
    lines = []
    legend_entries = []

    for group in unique_groups:
        # Create a custom legend entry for the group name (without a marker)
        legend_entries.append(Line2D([0], [0], color='none', label=group))
        
        # Plot each line that belongs to the current group and add to the legend
        for i in range(len(yvalues)):
            legend_label = legend_labels[i]
            if grouping_rule.get(legend_label) == group:
                line, = ax.plot(
                    xvalues[i], yvalues[i],
                    color=LINE_COLORS[i % len(LINE_COLORS)],  # Cycle through colors if more lines than colors
                    linewidth=LINE_WIDTH, marker=MARKERS[i], markersize=MARKER_SIZE,  # Adjust line width, marker, and size
                    label=legend_label, markeredgecolor='k'
                )
                lines.append(line)
                # Add the line to the legend entries
                legend_entries.append(line)

    # Add a legend if allowed
    if allow_legend:
        # Combine custom group labels with actual line labels
        all_legend_lines = legend_entries
        all_legend_labels = [entry.get_label() for entry in legend_entries]

        # Create the legend
        custom_legend = plt.legend(
            all_legend_lines, all_legend_labels,
            prop=LEGEND_FP,
                   loc='upper center',
                   ncol=1,
                   bbox_to_anchor=(-0.37, 1.0), shadow=False,
                   columnspacing=0.1,
                   frameon=True, borderaxespad=0, handlelength=1.2,
                   handletextpad=0.1,
                   labelspacing=0.1
              # Adjust this value to control the overall font size of the legend
        )

        # Apply custom styling to group labels (bold italics in red)
        for text, entry in zip(custom_legend.get_texts(), legend_entries):
            if isinstance(entry, Line2D) and entry.get_color() == 'none':
                text.set_fontsize(LEGEND_FONT_SIZE)  # Adjust group tag font size here if needed
                text.set_fontweight('bold')
                text.set_style('italic')
            else:
                # Apply default styling for line labels if needed
                text.set_fontsize(LEGEND_FONT_SIZE)  # Adjust line label font size here if needed
                text.set_color('black')

    plt.xlabel(x_label, fontproperties=LABEL_FP)
    plt.ylabel(y_label, fontproperties=LABEL_FP)
    plt.xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10))

    # Add a grid for better readability
    plt.grid(axis='y', color='gray')

    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    ax.get_xaxis().set_tick_params(direction='in', pad=10)
    ax.get_yaxis().set_tick_params(direction='in', pad=10)

    # Save the figure as a PDF
    plt.savefig(filename + ".pdf", bbox_inches='tight')
    plt.close(fig)  # Close the figure to fre