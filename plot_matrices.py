import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from representatives import difference_layers

with open("pickled_weights1.txt", "rb") as fp:   # Unpickling
    weights = pickle.load(fp)

for idx in range(4):
    min_list = [None]*len(weights)
    max_list = [None]*len(weights)
    
    for j in range(5):
        min_list[j] = np.amin(weights[j][2*idx])
        max_list[j] = np.amax(weights[j][2*idx])

    minimum = min(min_list)
    maximum = max(max_list)

    if idx%2 == 1:
        fig = plt.figure()
        plt.tight_layout()
        
        ax = fig.add_subplot(511)
        cax = ax.matshow(weights[0][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(0+1,idx+1))
        # fig.colorbar(cax)
        
        ax = fig.add_subplot(512)
        cax = ax.matshow(weights[1][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(1+1,idx+1))
        # fig.colorbar(cax)
        
        ax = fig.add_subplot(513)
        cax = ax.matshow(weights[2][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(2+1,idx+1))
        # fig.colorbar(cax)
        
        ax = fig.add_subplot(514)
        cax = ax.matshow(weights[3][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(3+1,idx+1))
        # fig.colorbar(cax)

        ax = fig.add_subplot(515)
        cax = ax.matshow(weights[4][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(4+1,idx+1))
        # fig.colorbar(cax)
    else:
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.add_subplot(151)
        cax = ax.matshow(weights[0][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(0+1,idx+1))
        # fig.colorbar(cax)
        
        ax = fig.add_subplot(152)
        cax = ax.matshow(weights[1][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(1+1,idx+1))
        # fig.colorbar(cax)
        
        ax = fig.add_subplot(153)
        cax = ax.matshow(weights[2][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(2+1,idx+1))
        # fig.colorbar(cax)
                
        ax = fig.add_subplot(154)
        cax = ax.matshow(weights[3][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(3+1,idx+1))
        # fig.colorbar(cax)
                
        ax = fig.add_subplot(155)
        cax = ax.matshow(weights[4][2*idx], cmap='RdBu', vmin = minimum, vmax = maximum)
        plt.xticks([])
        plt.yticks([])
        plt.title("Network: {}, Layer: {}".format(4+1,idx+1))
        # fig.colorbar(cax)
        
pdf = matplotlib.backends.backend_pdf.PdfPages("weight_matrices_5.pdf")
for fig in xrange(1, plt.gcf().number+1):
    pdf.savefig( fig )
pdf.close()

plt.show()

print "L2 Distance Between Network Layers"
print "__________"
print "Layer 1: (N1,N2) , (N1,N3) , (N2,N3)"
print [difference_layers(weights[0][0], weights[1][0]), difference_layers(weights[0][0], weights[2][0]), difference_layers(weights[1][0], weights[2][0])]

print "__________"
print "Layer 2: (N1,N2) , (N1,N3) , (N2,N3)"

print [difference_layers(weights[0][2], weights[1][2]), difference_layers(weights[0][2], weights[2][2]), difference_layers(weights[1][2], weights[2][2])]
print "__________"
print "Layer 3: (N1,N2) , (N1,N3) , (N2,N3)"

print [difference_layers(weights[0][4], weights[1][4]), difference_layers(weights[0][4], weights[2][4]), difference_layers(weights[1][4], weights[2][4])]
print "__________"
print "Layer 4: (N1,N2) , (N1,N3) , (N2,N3)"

print [difference_layers(weights[0][6], weights[1][6]), difference_layers(weights[0][6], weights[2][6]), difference_layers(weights[1][6], weights[2][6])]
