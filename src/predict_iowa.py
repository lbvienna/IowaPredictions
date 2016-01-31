import utilities
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, date

import warnings
warnings.filterwarnings('ignore')

eight_order = ['Huckabee', 'McCain', 'Romney', 'Thompson', 'Paul', 'Giuliani']
twelve_order = ['Paul', 'Romney', 'Santorum', 'Gingrich', 'Perry', 'Bachmann', 'Huntsman', 'Cain']
sixteen_order = ['Trump', 'Cruz', 'Rubio', 'Carson', 'Bush', 'Christie', 'Paul', 'Kasich', 'Huckabee', 'Fiorina', 'Santorum']
twelve_endorsements = np.asarray([0.97669759, 0.68541745, 0.46404455, 0.84853433])

pollster_grades = {'A+': 10, 'A': 9, 'A-': 8, 'B+': 7, 'B': 6, 'B-': 5, 'C+': 2, 'C': 1.5, 'C-': 1, 'D+': 0.5, 'D': 0.3, 'D-': 0.1, 'F':0}


def main():
	year = "2016"
	if year == '2016':
		tp = 6.36636636637
		alpha = 3.01101101101
		X_twelve, Y_twelve = pre_processing(year='2012', time_penalty=tp)
		X_eight, Y_eight = pre_processing(year='2008', time_penalty=tp)
		X_train = np.concatenate((X_twelve, X_eight))
		Y_train = np.concatenate((Y_twelve, Y_eight))
		X_sixteen, ___ = pre_processing(year=year, time_penalty=tp)
		predictions = run_classifier(X_train, Y_train, X_sixteen, alpha=alpha)
		print "Predictions on {0}".format(date.today())
		results = []
		for i, name in enumerate(sixteen_order):
			results.append((name, (predictions[i]/sum(predictions))*100))
		sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
		for i, (name, result) in enumerate(sorted_results):
			print "		{0}: {1:.2f}".format(name, result)
	else:
		tp = 6.36636636637
		X_twelve, Y_twelve = pre_processing(year='2012', time_penalty=tp)
		X_eight, Y_eight = pre_processing(year='2008', time_penalty=tp)
		X = np.concatenate((X_twelve, X_eight))
		Y = np.concatenate((Y_twelve, Y_eight))
		best_alpha = find_alpha(X, Y)
		mse = run_test_classifier(X, Y, best_alpha)	

def optimize_time_penalty():
	all_mse = []
	tps = np.linspace(0, 8, num=1000)
	for tp in tps:
		X_twelve, Y_twelve = pre_processing(year='2012', time_penalty=tp)
		X_eight, Y_eight = pre_processing(year='2008', time_penalty=tp)
		X = np.concatenate((X_twelve, X_eight))
		Y = np.concatenate((Y_twelve, Y_eight))
		best_alpha = 3.01101101101
		coef, mse = run_test_classifier(X, Y, best_alpha)
		all_mse.append(mse)
	min_mse_index = np.argmin(all_mse)
	print "Minimum MSE is {0} with Time Penalty of {1}".format(all_mse[min_mse_index], tps[min_mse_index])
	plt.plot(tps, all_mse)
	plt.xlabel("Time Penalties")
	plt.ylabel("MSE")
	plt.title("MSE vs Time Penalties for Weighted Poll Average")
	plt.show()

def optimize_endorsements():
	endorsements_vector = np.asarray([1, 1, 1, 1]).astype(float)
	endorsements_matrix = np.asarray([2, 27, 46, 13]).astype(float)
	best_alpha = 3.01101101101
	tp = 6.37437437437
	for i in xrange(1000):
		old_vec = np.copy(endorsements_vector)
		X_twelve, Y_twelve = pre_processing(year='2012', time_penalty=tp, endorsements_vector=endorsements_vector)
		X_eight, Y_eight = pre_processing(year='2008', time_penalty=tp, endorsements_vector=endorsements_vector)
		X = np.concatenate((X_twelve, X_eight))
		Y = np.concatenate((Y_twelve, Y_eight))
		coef, mse = run_test_classifier(X, Y, best_alpha)
		a = (-0.01) * (coef[1]/float(X.shape[0]))
		endorsements_vector +=  a * endorsements_matrix
		print mse
		print endorsements_vector

def run_classifier(X_train, Y_train, X_real, alpha):
	clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
	clf.fit(X_train, Y_train)
	predictions = clf.predict(X_real)
	plot_classification(X_train, Y_train, X_real, predictions, clf.coef_)
	return predictions

def plot_classification(X_train, Y_train, X_real, predictions, coefficients):
	tuples = [(X_train[:,1][i], Y_train[i][0]) for i in xrange(len(Y_train))]
	sorted_tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
	training = plt.scatter(*zip(*sorted_tuples))
	plt.title("Polls vs actual results")
	plt.xlabel("Weighted Polls")
	plt.ylabel("Actual Iowa Caucus Result")
	axes = plt.gca()
	axes.set_xlim([0, 40])
	axes.set_ylim([0, 40])
	x = range(0, 40)
	y = [coefficients[1] * x_i for x_i in x]
	plt.plot(x, y)
	sixteen_tuples = [(X_real[:,1][i], predictions[i]) for i in xrange(len(X_real))]
	sorted_sixteen_tuples = sorted(sixteen_tuples, key=lambda x: x[0], reverse=True)
	real = plt.scatter(*zip(*sorted_sixteen_tuples), color='red')
	plt.legend([training, real], ['2008 and 2008 training', '2016 predictions'])
	plt.show()
	print sorted_tuples


def run_test_classifier(X, Y, alpha):
	mse, r2, coef= loocv(X, Y, alpha = alpha)
	print "MSE is {0}".format(mse)
	print "R2 is {0}".format(r2)
	return coef, mse


def find_alpha(X, Y):
	alphas = np.linspace(0, 8, num=1000)
	all_mse = []
	all_r2 = []
	for alpha in alphas:
		mse, r2, __ = loocv(X, Y, alpha=alpha)
		all_mse.append(mse)
		all_r2.append(r2)
	min_mse_index = np.argmin(all_mse)
	max_r2_index = np.argmax(all_r2)
	print "Minimum MSE is {0} with alpha of {1}".format(all_mse[min_mse_index], alphas[min_mse_index])
	print "Maximum R2 is {0} with alpha of {1}".format(all_r2[max_r2_index], alphas[max_r2_index])
	plt.plot(alphas, all_mse)
	plt.xlabel("Different Values of Alpha")
	plt.ylabel("MSE with LOOCV")
	plt.title("MSE over different values of Alpha")
	plt.show()
	return alphas[min_mse_index]

def loocv(X, Y, alpha=0.1):
	coefficients = np.zeros_like(X[0])
	predictions = []
	y_actuals = []
	for i in xrange(len(Y)):
		X_i = np.delete(X, i, axis=0)
		Y_i = np.delete(Y, i)
		x_i = X[i]
		y_i = Y[i]
		prediction, y_actual, coef = test_classify(X_i, Y_i, x_i, y_i, alpha=alpha)
		predictions.append(prediction)
		y_actuals.append(y_actual)
		coefficients += coef
	print coefficients / len(Y)
	mse = mean_squared_error(y_actuals, predictions)
	r2 = r2_score(y_actuals, predictions)
	return mse, r2, coef

def test_classify(X_train, Y_train, X_test, Y_test, alpha):
	clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
	clf.fit(X_train, Y_train)
	prediction = clf.predict(X_test)
	return prediction, Y_test, clf.coef_

def pre_processing(year="2012", time_penalty=3, endorsements_vector=[0.97669759, 0.68541745, 0.46404455, 0.84853433]):
	polling_data, fundamentals, results = get_data(year)
	x, y = combine_data(polling_data, fundamentals, results, year, time_penalty, endorsements_vector, polls_only=True)
	return x, y

def combine_data(polling_data, fundamentals, results, year, time_penalty, endorsements_vector, polls_only=True):
	X = []
	y = []
	if year == '2012':
		names = twelve_order
	elif year == '2008':
		names = eight_order
	else:
		names = sixteen_order
	for name in names:
		weighted_average, non_weighted_average = weight_polling(polling_data[name], year, polling_data['Date'], polling_data['Poll'], time_penalty)
		if polls_only is False:
			endorsements = np.asarray(fundamentals[name][0:4]).dot(endorsements_vector)
			experience = np.asarray(fundamentals[name][4:10]).dot(twelve_experience)
			X.append(np.asarray([weighted_average, endorsements]))
		else:
			X.append(np.asarray([1, weighted_average]))
		if year == '2012' or year == '2008':
			y.append(results[name])
	return np.asarray(X), np.asarray(y).reshape(-1, 1)

def weight_polling(polling_data, year, polling_date, polling_org, time_penalty):
	pollster_information = utilities.read_csv('pollster-ratings_{0}.tsv'.format(year), year=year, data_values=False, delim='\t')
	weighted_average = 0
	weights_summed = 0
	non_weighted_average = 0
	num_polls = 0
	for i, polling_result in enumerate(polling_data):
		pollster = polling_org[i]
		if pollster == 'RCP Average' or pollster == 'Final Results':
			continue
		polling_time = polling_date[i]
		difference_date = get_date_difference(polling_time, year)
		if difference_date is not None:
			if pollster in pollster_information.keys():
				letter_grade = pollster_information[pollster][9]
			else:
				letter_grade = 'C+'
			if letter_grade in pollster_grades:
				numerical_grade = pollster_grades[letter_grade]
				weighted_average += ((1/float(abs(difference_date.days)**time_penalty)) * numerical_grade * polling_result)
				non_weighted_average += polling_result
				weights_summed += (numerical_grade * (1/float(abs(difference_date.days)**time_penalty)))
				num_polls += 1
	return weighted_average / float(weights_summed), non_weighted_average/ float(num_polls)

def get_date_difference(polling_time, year):
	if len(polling_time) > 4:
		hyphen = polling_time.find('-')
		polling_time = polling_time[:hyphen-1]
	else:
		return
	if year == '2012':
		if polling_time == '1/1':
			polling_time += '/2012'
		else:
			polling_time += '/2011'
	elif year == '2016':
		if polling_time[:2] == '1/':
			polling_time += '/2016'
		else:
			polling_time += '/2015'
	elif year == '2008':
		if polling_time == '1/1':
			polling_time += '/2008'
		else:
			polling_time += '/2007'
	else:
		raise "No year specified"
	datetime_poll = datetime.strptime(polling_time, "%m/%d/%Y")
	if year == '2012':
		iowa_date = datetime.strptime("1/3/2012", "%m/%d/%Y")
	elif year == '2008':
		iowa_date = datetime.strptime("1/3/2008", "%m/%d/%Y")
	else:
		#iowa_date = datetime.strptime("2/1/2016", "%m/%d/%Y")
		iowa_date = datetime.strptime("1/31/2016", "%m/%d/%Y")
	difference = datetime_poll - iowa_date
	return difference


def get_data(year):
	polling_data, results = utilities.read_csv("{0}_all_polling.csv".format(year), year)
	fundamentals = utilities.read_csv("{0}_fundamentals.csv".format(year), year)
	return polling_data, fundamentals, results

if __name__ == '__main__':
	main()