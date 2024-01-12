import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.pylab as pl 
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection 
from sklearn import cluster, covariance, manifold


#Define a function to calculeta max drawdown
def mdd(x):
    #x is a return vector
    wealth = (x+1).cumprod()
    #determine cumulative maximum value
    cummax = wealth.cummax()
    #calculate drawdown vector
    drawdown = wealth/cummax - 1
    return drawdown.min()

#Summary statistics for assets
def getSumStat(data, rounding = 2):
    # Get the Start and End date of the dataset
    date_obj = data.index[0]
    start_of_week = date_obj - timedelta(days=date_obj.weekday())
    start = start_of_week.strftime("%m/%d/%Y")
    end =  data.index[-1].strftime("%m/%d/%Y")
    
    print('Summary Statistic Information from ' + start + ' to ' + end + ':')
    if(data.isnull().values.any()):
        print('WARNING: Some assets have missing data during this time period!')
        print('Dropping assets: ')
        for Xcol_dropped in list(data.columns[data.isna().any()]): print(Xcol_dropped)
        data = data.dropna(axis='columns')
    
    ss_temp = pd.DataFrame(index = data.columns)
    ss_temp['Total Return(%)'] = np.round((((data+1).cumprod()-1)*100).iloc[-1] , rounding)
    ss_temp['Ave Return(%)'] = np.round(data.mean()*100, rounding)
    ss_temp['Annu. Ave Return(%)'] = np.round(((data.mean()+1)**52-1)*100, rounding)
    ss_temp['Annu. Std(%)'] = np.round(data.std()*np.sqrt(52)*100, rounding)
    ss_temp['Max Drawdown(%)'] = np.round(data.apply(mdd)*100, rounding)
    return(ss_temp)


#Graphical analysis code
def graphicalAnalysis(dataset, start_date, end_date, display_SumStat=True):
    # Check if the input dates are legitimate
    if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
        raise ValueError('ERROR: The entered "start_date" should be before "end_date".')

    if dataset.index[0] - timedelta(days=dataset.index[0].weekday()) > datetime.strptime(start_date, "%Y-%m-%d"):
        print('WARNING: The entered "start_date" is outside of the range for the given dataset.')
        print('The "start_date" is adjusted to the earliest start_date, i.e.',
              (dataset.index[0] - timedelta(days=dataset.index[0].weekday())).strftime("%Y-%m-%d"))
        print()

    if dataset.index[-1] < datetime.strptime(end_date, "%Y-%m-%d"):
        print('WARNING: The entered "end_date" is outside of the range for the given dataset.')
        print('The "end_date" is adjusted to the latest end_date, i.e.',
              dataset.index[-1].strftime("%Y-%m-%d"))
        print()

    # Extract data for the current time period
    temp = dataset[dataset.index >= start_date].copy()
    X = temp[temp.index <= end_date].copy()

    # Check if there is NA in the dataset within the given time period
    # If yes, then drop those assets before doing graphical analysis
    if X.isnull().values.any():
        print('WARNING: Some assets have missing data during this time period!')
        print('Dropping assets:')
        for Xcol_dropped in list(X.columns[X.isna().any()]):
            print(Xcol_dropped)
        X = X.dropna(axis='columns')
        print()

    # Get start and end date in dataset
    date_obj = X.index[0]
    start_of_week = date_obj - timedelta(days=date_obj.weekday())
    start = start_of_week.strftime("%m/%d/%Y")
    end = X.index[-1].strftime("%m/%d/%Y")

    # Get asset names in dataset
    names = np.array(list(X.columns))

    # Show the number of assets examined
    print('Number of assets examined:', X.shape[1])

    # Estimate precision matrix using Graphical Lasso
    edge_model = covariance.GraphicalLassoCV(max_iter=1000)

    # Standardize the time series. The purpose is to use correlations rather than covariance as is more efficient to structure recovery
    X_std = X / X.std(axis=0)
    edge_model.fit(X_std)

    # Cluster using affinity propagation
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

    # Use manifold.MDS to project the information on a lower dimension 2D plane. Find the best positions for the nodes.
    node_position_model = manifold.MDS(n_components=2, random_state=0)
    embedding = node_position_model.fit_transform(X_std.T).T

    # Visualization I
    # Specify node colors by cluster labels
    color_list = pl.cm.jet(np.linspace(0, 1, n_labels + 1))
    my_colors = [color_list[i] for i in labels]

    # Compute partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Extract edge values based on the non-zero partial correlations as masked above
    values = np.abs(partial_correlations[non_zero])
    val_max = values.max()

    # Title of the plot. Note that we have resampled dataset to weekly frequencies but pls feel free to change it
    title = 'Graphical Network Analysis of Selected Tickers over the Period '+start+' to '+end+' (Weekly)'

    #Display
    graphicalAnalysis_plot(d, partial_correlations, my_colors,
                           names, labels, embedding, val_max, title)
    
    # The configuration of the plot
    plot_config = [d, partial_correlations, my_colors, names, labels, embedding, val_max, title]

    # Show summary statistics for each firm over the given period
    if (display_SumStat):
        display(getSumStat(X))


# Function Used for plotting the graphical network graph
def graphicalAnalysis_plot(d, partial_correlations, my_colors,
                           names, labels, embedding, val_max, title):
        
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    n_labels = labels.max()
    
    #For correlation network graph
    fig = plt.figure(1, facecolor='w', figsize=(12, 5))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=500 * d ** 2, c= my_colors)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r, 
                        norm=plt.Normalize(0, .7 * val_max))
    lc.set_array(values)
    temp = (15 * values)
    temp2 = np.repeat(5, len(temp))
    w = np.minimum(temp, temp2)
    lc.set_linewidths(w)
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label('Strength')

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())
    plt.title(title)
    plt.show()
# END of function graphicalAnalysis_plot
# #############################################################################