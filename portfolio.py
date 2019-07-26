import pandas as pd
import numpy as np
import unittest

class PortfolioPrice:
	'''Class for calculate the prise of asset.
	
	Class has instance variable:
	prices -- daily asset prices
	weights -- daily portfolio weights

	Methods:
	calculate_asset_performance -- calculate asset portfolio performance

	'''
	def __init__(self, prices, weights):
		'''Сlass constructor. Define instance variable'''
		self.prices = prices
		self.weights = weights

	def calculate_asset_performance(self, start_date, end_date):
		'''Method to calculate price asset portfolio performance.
		
		Keyword arguments:
		start_date -- start date for the calculation
		end_date -- end date for the calculation
		
		Return:
		asset -- pandas.Series with portfolio performances in the specified
		interval

		'''
		interval = self._get_interval(start_date, end_date, self.prices)
		asset = pd.Series(0, index=interval)
		asset.loc[interval[0]] = 1
		for i in range(1, len(interval)):
			asset.loc[interval[i]] = asset.loc[interval[i-1]] * (1 + 
									 self._with_weight_per_day(
									 	interval[i], self._asset_per_two_day))
		return asset

	def _asset_per_two_day(self, date):
		'''Method to calculate price asset at time 'date'.

		Keyword arguments:
		date -- date for which the calculation should be made

		Return:
		asset price at time 'date' for all assets as np.array

		'''
		price = np.array([])
		#Get index of specified date in prices
		index = self.prices.loc[self.prices[
									self.prices.columns[0]] == date].index[0]
		seq = (index - 1, index) if index > 0 else (index, index)
		for i in seq:
			price = np.append(price, self._find_valid_value(i, self.prices))
		return price.reshape(2, len(self.prices.columns[1:]))
	
	def _with_weight_per_day(self, date, method):
		'''Method to calculate asset with weights.

		Keyword arguments:
		date -- date for which the calculation should be made
		method - for which assets should be calculate

		Return:
		res -- weighted sum specified asset
		
		'''
		names = self.prices.columns[1:]
		par = method(date)
		res = 0
		weight = self._get_weight(date)
		for i in range(len(names)):
			res += weight[i] * ((par[1][i] - par[0][i]) / par[0][i])
		return res

	def _get_weight(self, date):
		'''Method to obtain weight for a specified date.

		Keyword arguments:
		date -- date for which the calculation should be made
		
		Return:
		weight of the asset for a specified date with valid value

		'''		
		weight = np.array([])
		date = self._get_valid_date(date, self.weights)
		#Get index of specified date in weights
		index = self.weights.loc[self.weights[
								 	self.weights.columns[0]] == date].index[0]
		return self._find_valid_value(index, self.weights)

	def _find_valid_value(self, index, element):
		'''Method to check for a valid value.

		Keyword arguments:
		index -- index of the value to check
		element -- element in which need to check

		Return:
		valid values of the assets element by index

		'''			
		valid_index = index
		temp = element.loc[index][1:].copy()
		while valid_index > 0 and any(temp.isna()):
			valid_index -= 1
			for col in temp.index:
				if temp.isna()[col]:
					temp[col] = element.loc[valid_index - 1][col]
		valid_index = index
		while any(temp.isna()) and valid_index < len(element):
			valid_index += 1
			for col in temp.index:
				if temp.isna()[col]:
					temp[col] = element.loc[valid_index - 1][col]
		return temp.values

	def _get_valid_date(self, date, element):
		'''Method to check for a valid date.

		Keyword arguments:
		date -- date to check
		element -- element in which need to check

		Return:
		valid date of element

		'''		
		if date in element[element.columns[0]].values:
			return date
		temp = np.append(element[element.columns[0]].values, date)
		temp.sort()
		index = np.where(temp == date)[0][0]
		index = index - 1 if index > 0 else index + 1
		return element.loc[index][element.columns[0]]

	def _get_interval(self, start_date, end_date, element):
		'''Method to get for a valid interval of date.

		Keyword arguments:
		start_date -- start date for interval
		end_date -- end date for interval
		element -- element in which need to check

		Return:
		valid date interval of element

		'''			
		all_date = element[element.columns[0]].values
		start_date = self._get_valid_date(start_date, element)
		end_date = self._get_valid_date(end_date, element)
		start = np.where(all_date == start_date)[0][0]
		end = np.where(all_date == end_date)[0][0]
		return all_date[start: end + 1]	
	

class PortfolioCurrency(PortfolioPrice):
	'''Class for calculate the currency of asset.
	
	Class has instance variable:
	prices -- daily asset prices
	weights -- daily portfolio weights
	currencies -- asset currencyextend
	exchanges -- daily exchange rate (to dollar)

	Methods:
	calculate_currency_performance -- calculate currency portfolio performance

	'''
	def __init__(self, prices, weights, currencies, exchanges):
		'''Сlass constructor. Define instance variable.

		Method extend the superclass method (__init__)

		'''
		PortfolioPrice.__init__(self, prices, weights)
		self.currencies = currencies
		self.exchanges = exchanges

	def calculate_currency_performance(self, start_date, end_date):
		'''Method to calculate currency portfolio performance.
		
		Keyword arguments:
		start_date -- start date for the calculation
		end_date -- end date for the calculation
		
		Return:
		currency -- pandas.Series with portfolio performances in the specified
		interval

		'''
		interval = self._get_interval(start_date, end_date, self.exchanges)
		currency = pd.Series(0, index=interval)
		currency.loc[interval[0]] = 1
		for i in range(1, len(interval)):
			currency.loc[interval[i]] = currency.loc[interval[i-1]] * (1 + 
									 self._with_weight_per_day(
									 			interval[i], 
									 			self._currency_per_two_day))
		return currency

	def _currency_per_two_day(self, date):
		'''Method to calculate currency at time 'date'.

		Keyword arguments:
		date -- date for which the calculation should be made

		Return:
		currency at time 'date' for all assets as np.array

		'''
		currency = np.array([])
		names = self.prices.columns[1:]
		index = self.exchanges.loc[
				self.exchanges[self.exchanges.columns[0]] == date].index[0]
		seq = (index - 1, index) if index > 0 else (index, index)
		for i in seq:
			for name in names:
				if (self.currencies.loc[name] == 'USD').bool():
					currency = np.append(currency, 1)
				else:
					temp = self._find_valid_value(i, self.exchanges)
					cur_i = np.where(self.exchanges.columns[1:] == 
									 self.currencies.loc[name].values[0])[0][0]
					currency = np.append(currency, temp[cur_i])
		return currency.reshape(2, len(self.prices.columns[1:]))

class PortfolioTotal(PortfolioCurrency):
	'''Class for calculate the total price of asset.
	
	Class has instance variable:
	prices -- daily asset prices
	weights -- daily portfolio weights
	currencies -- asset currencyextend
	exchanges -- daily exchange rate (to dollar)

	Methods:
	calculate_total_performance -- calculate total portfolio performance

	'''
	def __init__(self, prices, weights, currencies, exchanges):
		'''Сlass constructor. Define instance variable.

		Method calls the superclass method (__init__)

		'''
		PortfolioCurrency.__init__(self, prices, weights, currencies,
								   exchanges)
			
	def calculate_total_performance(self, start_date, end_date):
		'''Method to calculate total portfolio performance.
		
		Keyword arguments:
		start_date -- start date for the calculation
		end_date -- end date for the calculation
		
		Return:
		total -- pandas.Series with portfolio performances in the specified
		interval

		'''
		interval = self._get_interval(start_date, end_date, self.prices)
		total = pd.Series(0, index=interval)
		total.loc[interval[0]] = 1
		for i in range(1, len(interval)):
			total.loc[interval[i]] = total.loc[interval[i-1]] * (1 + 
									 self._with_weight_per_day(
									 interval[i], self._total_per_two_day))
		return total

	def _total_per_two_day(self, date):
		'''Method to calculate total at time 'date'.

		Keyword arguments:
		date -- date for which the calculation should be made

		Return:
		total at time 'date' for all assets as np.array

		'''
		price = self._asset_per_two_day(date)
		date = self._get_valid_date(date, self.exchanges)
		currency = self._currency_per_two_day(date)
		return price * currency


class TestPortfolioValue(unittest.TestCase):
	'''Class to check the value of the portfolio.'''
	def test_first_value(self):
		'''Method to check the first value of portfolio.'''
		portfolio_price = PortfolioPrice(prices, weights)
		portfolio_currency = PortfolioCurrency(prices, weights, 
											   currencies, exchanges)
		portfolio_total = PortfolioTotal(prices, weights, 
										 currencies, exchanges)
		#Getting full date interval
		interval_date = prices[prices.columns[0]].values
		start_date = interval_date[0]
		end_date = interval_date[-1]
		middle_date = interval_date[len(interval_date)//2]
		#Result list of portfolio performance
		result = []
		for start, end in [(start_date, end_date),
						   (start_date, middle_date),
						   (middle_date, end_date)]:
			result.append(portfolio_price.calculate_asset_performance(
				start, end))
			result.append(portfolio_currency.calculate_currency_performance(
				start, end))
			result.append(portfolio_total.calculate_total_performance(
				start, end))
		for portfolio in result:
			check_value = portfolio.loc[portfolio.index[0]]
			self.assertEqual(check_value, 1)

if __name__ == '__main__':
	prices = pd.read_csv('prices.csv')
	currencies = pd.read_csv('currencies.csv', index_col='Unnamed: 0')
	weights = pd.read_csv('weights.csv')
	exchanges = pd.read_csv('exchanges.csv')

	suite = unittest.TestLoader().loadTestsFromTestCase(TestPortfolioValue)
	unittest.TextTestRunner(verbosity=2).run(suite)
