using PythonPlot

function plotimages(x, y, suptitle; nrows = 5, ncols = 10)
    fig, axes = subplots(nrows, ncols, sharex = true, sharey = true, dpi = 120)
    for (i, idx) in enumerate(CartesianIndices((nrows, ncols)))
        ax = axes[idx[1]-1,idx[2]-1]
        ax.imshow(x[:,:,1,i]')
        ax.set(title = y[i])
        ax.get_xaxis().set_visible(false)
        ax.get_yaxis().set_visible(false)
    end
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
end
