import numpy as np
import random
import Real_Data_Process as RDP


def RPVA(ldp_protocol, domain, origin_data):
	"""
	Implements the Random Perturbed-value Attack (RPVA).
	Args:
		ldp_protocol (str): 'GRR' for discrete data, 'Laplace' for continuous data.
		domain (list): The domain of the data (discrete or continuous).
		origin_data (list): the value of origin data
		epsilon (float): Privacy budget for Laplace mechanism.
	Returns:
		The randomly chosen perturbed value.
	"""
	if ldp_protocol == 'GRR':
		disturb_value = []
		while True:
			for i in origin_data:
				temp_domain = domain.copy()
				temp_domain.remove(i)
				disturb_value.append(random.choice(temp_domain))
			return disturb_value
	else:
		while True:
			disturb_value = [random.uniform(domain[0], domain[1]) for _ in origin_data]
			if all(item not in origin_data for item in disturb_value):
				return disturb_value


def RIA(ldp_protocol, domain, origin_data, epsilon=None, input_domain=None):
	"""
	Implements the Random Item Attack (RIA).
	Args:
		domain (list): The domain of the data (discrete or continuous).
		ldp_protocol (str): 'GRR' for discrete data, 'Laplace' for continuous data.
		origin_data (list): the value of origin data with LDP but unattacked
		epsilon (float): Privacy budget for Laplace mechanism.
	Returns:
		The perturbed value after RIA.
	"""
	if ldp_protocol == 'GRR':
		disturb_value = []
		for i, item in enumerate(origin_data):
			while True:
				temp_domain = input_domain.copy()
				if item in temp_domain and len(temp_domain) != 1:
					temp_domain.remove(origin_data[i])  # 保证输入数据不一样
				temp_data = RDP.grr_mechanism(random.choice(temp_domain), domain, epsilon)
				if temp_data != item:
					disturb_value.append(temp_data)
					break
		return disturb_value
	else:
		while True:
			result_data = [RDP.laplace_mechanism(random.uniform(domain[0], domain[1]), epsilon) for _ in origin_data]
			if all(item not in origin_data for item in result_data):
				return result_data


def RPA(true_value, origin_data, ldp_protocol, domain=None):
	"""
	Implements the Random Parameter Attack (RPA) by altering LDP parameters.
	Args:
		true_value (list): True value
		ldp_protocol (str): 'GRR' or 'Laplace'.
		origin_data (list): the value of origin data
		domain (list): The domain of the data (discrete or continuous).
	Returns:
		dict: Modified parameters.
	"""
	epsilon = round(random.uniform(0.1, 1), 3)  # Randomly adjust epsilon
	if ldp_protocol == 'GRR':
		disturb_value = []
		for i, item in enumerate(origin_data):
			while True:
				temp_data = RDP.grr_mechanism(true_value[i], domain, epsilon)
				if temp_data != item:  # 保证攻击结果
					disturb_value.append(temp_data)
					break
		return disturb_value
	else:
		while True:
			attacked_values = [RDP.laplace_mechanism(i, epsilon) for i in true_value]
			if all(item not in origin_data for item in attacked_values):
				return attacked_values


def MGA(target_item, ldp_protocol, origin_data, domain=None, epsilon=None):
	"""
	Implements the Maximal Gain Attack (MGA).
	Args:
		domain (list): The domain of the data (discrete or continuous).
		ldp_protocol (str): 'GRR' for discrete data, 'Laplace' for continuous data.
		epsilon (float): Privacy budget for Laplace mechanism.
	Returns:
		The perturbed value after MGA.
	"""
	if ldp_protocol == 'GRR':
		disturb_value = []
		for i in origin_data:
			while True:
				attacked_values = RDP.grr_mechanism(target_item, domain, epsilon)
				if attacked_values != i:
					disturb_value.append(attacked_values)
					break
		return disturb_value
	else:
		while True:
			attacked_values = [RDP.laplace_mechanism(target_item, epsilon) for _ in origin_data]
			if all(value not in origin_data for value in attacked_values):
				return attacked_values
