import numpy as np
import random
import Real_Data_Process as RDP


def DIPA(ldp_protocol, domain, origin_LDP_data, epsilon, ture_df_copy):
	"""
	Deliberate Input-Poisoning Attack
	Args:
		domain (list): The domain of the feature data (discrete or continuous).
		ldp_protocol (str): 'GRR' for discrete data, 'Laplace' for continuous data.
		origin_LDP_data (list): the value of origin data with LDP but unattacked in current time
		epsilon (float): Privacy budget for Laplace mechanism.
		input_domain (list): domain list of all user input data (LDP)
		bias_range (float): the bias range of attack
	Returns:
		The perturbed value
	"""
	disturb_value = []
	if ldp_protocol == 'GRR':
		# 找出满足个人域中离频数最高点最远的元素
		user_domain = ture_df_copy.unique().tolist()
		attack_result = max([d for d in user_domain if d in domain], key=lambda x: domain.index(x))
		for i, _ in enumerate(ture_df_copy):
			while True:
				temp_data = RDP.grr_mechanism(attack_result, domain, epsilon)
				if temp_data != origin_LDP_data[i]:
					disturb_value.append(temp_data)
					break
		return disturb_value
	
	else:
		while True:
			for _ in range(len(ture_df_copy)):
				disturb_value.append(RDP.laplace_mechanism(domain[0], epsilon))
			return disturb_value


def DPPA(true_value, origin_data, ldp_protocol, domain=None):
	"""
	Deliberate Parameter-Poisoning Attack
	Args:
		true_value (list): True value
		ldp_protocol (str): 'GRR' or 'Laplace'.
		origin_data (list): the value of origin data
		domain (list): The domain of the data (discrete or continuous).
	Returns:
		dict: Modified parameters.
	"""
	
	random_numbers = [random.randint(1, 1000) for _ in range(len(true_value))]
	epsilon_list = [num / sum(random_numbers) * 32 for num in random_numbers]
	disturb_value = []

	if ldp_protocol == 'GRR':
		for i, item in enumerate(origin_data):
			while True:
				temp_data = RDP.grr_mechanism(true_value[i], domain, epsilon_list[i])
				if temp_data != item:  # 保证攻击结果
					disturb_value.append(temp_data)
					break
		return disturb_value
	else:
		while True:
			for i, item in enumerate(true_value):
				temp_data = RDP.laplace_mechanism(item, epsilon_list[i])
				disturb_value.append(temp_data)
			return disturb_value


def ROPA(ldp_protocol, domain, origin_data):
	"""
	Randomized Output-Poisoning Attack
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
			return disturb_value


def SMGPA(target_item, ldp_protocol, origin_data, domain=None, epsilon=None):
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
			return attacked_values
