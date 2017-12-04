import os
import numpy as np
import pandas as pd

from biom import load_table
from itertools import compress
import csv


def main():
	table = load_table('C:/Users/Name/Documents/GitHub/forensic-geolocation/data/processed/global/100ITS.biom').pa()
	eliminate_exceptionally_rare_taxa = lambda values, id_, md: np.mean(values > 0) > 0
	table.filter(eliminate_exceptionally_rare_taxa, axis='observation')

	L = table.metadata_to_dataframe('sample')
	allID  = list(L.index)
	country = L['country']
	#latlon = L[['lat','lon']]

	for _ in table.ids('observation'):
		neededIDs = list(compress(allID, table.data(_,'observation') ))
		#neededLatLon = latlon.loc[neededIDs]
		neededCountry = country.loc[neededIDs]
		distinctC = list(set(neededCountry))
		#if (len(distinctC)>5 and len(distinctC)<12):
		if (len(distinctC)<3):
			with open(os.path.join(os.getcwd(),'locs101',_+'countries.csv'),'w') as mycsv:
				wr = csv.writer(mycsv, quoting=csv.QUOTE_ALL)
				wr.writerow(distinctC)

if __name__ == '__main__':
	main()