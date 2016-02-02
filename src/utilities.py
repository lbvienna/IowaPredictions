import csv
import os
import numpy as np

eight_order = ['Huckabee', 'McCain', 'Romney', 'Thompson', 'Paul', 'Giuliani']

twelve_order = ['Paul', 'Romney', 'Santorum', 'Gingrich', 'Perry', 'Bachmann', 'Huntsman', 'Cain']

sixteen_order = ['Trump', 'Cruz', 'Rubio', 'Carson', 'Bush', 'Christie', 'Paul', 'Kasich', 'Huckabee', 'Fiorina', 'Santorum']



def read_csv(filename, year, data_values=True, delim=','):
	data = {}
	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=delim)
		for row in reader:
			if data_values:
				try:
					data[row[0]] = [float(i) for i in row[1:]]
				except:
					data[row[0]] = row[1:]
			else:
				data[row[1]] = row
	if 'polling' in filename:
		if '2012' in filename:
			data = reshape_polling(data, year)
		else:
			data = reshape_polling(data, year)
	return data


def reshape_polling(polling_data, year):
	election = {}
	data = {}
	if year == '2012':
		names = twelve_order
	elif year == '2008':
		names = eight_order
	else:
		names = sixteen_order
	for i, (key, value) in enumerate(polling_data.iteritems()):
		if key in names:
			floats_value = []
			sum_ = 0.0
			nums = 0.0
			for j in value[2:]:
				try:
					float_i = float(j)
					sum_ += float_i
					nums += 1
				except ValueError as e:
					float_i = '---'
				floats_value.append(float_i)
			new_floats_value = []
			for j in floats_value:
				try:
					float_i = float(j)
				except ValueError as e:
					float_i = sum_/float(nums)
				new_floats_value.append(float_i)
			data[key] = new_floats_value
			if year == '2012' or year == '2008':
				try:
					election[key] = float(value[0])
				except:
					election[key] = 0.0
		else:
			data[key] = value
	return data, election
	