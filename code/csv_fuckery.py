import re
import os
import csv


for fil in os.listdir('../data/train_data'):


	with open('../data/train_data/' + fil, 'rb') as f:
		data = f.readlines()
		print data[0]

		for i in range(len(data)):
			data[i] = re.sub( '\s+', ',', data[i].lstrip().strip()) + '\n'

		with open('../data/corrected_train_data/' + fil, 'wb') as g:
			g.writelines(data)

		# csvwrite = csv.writer(g, delimiter=',')
		
		# for row in data:
		# 	print row
		# 	csvwrite.writerow(row)