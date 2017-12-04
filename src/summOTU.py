import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from biom import load_table
import matplotlib.pyplot as plt


def main():
        '''This file computes a histogram of the log-transformed
        OTU counts, for all taxa taken together and for an individual taxon,
        from the 100% OTU .biom file.'''
        
        filepath = 'C:/Users/Name/Documents/GitHub/forensic-geolocation/data/processed/global/100ITS.biom'
	origtable = load_table(filepath)
	#table = origtable.pa()
	#eliminate_exceptionally_rare_taxa = lambda values, id_, md: np.mean(values > 0) > 0
	#table.filter(eliminate_exceptionally_rare_taxa, axis='observation')
	#origtable.filter(eliminate_exceptionally_rare_taxa, axis='observation')

	print(origtable.get_table_density())
	mat = origtable.matrix_data

	(i,j,v) = sp.find(mat)
	print(max(v))
	plt.hist(np.log(v), bins=7)
	plt.title("Histogram of nonzero OTU counts (100% OTU)")
	plt.xlabel("Logarithm of Total OTU Counts")
	plt.ylabel("Frequency")
	plt.show()

	(i,j,v) = sp.find(mat[:,0])
	plt.hist(np.log(v), bins=7)
	plt.title("Histogram of nonzero OTU counts (100% OTU) for single OTU")
	plt.xlabel("Logarithm of Counts of Taxon 1 across all samples")
	plt.ylabel("Frequency")
	plt.show()

if __name__ == '__main__':
	main()
