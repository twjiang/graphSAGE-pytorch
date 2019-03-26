import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

def evaluate(dataCenter, ds, graphSage, classification, device):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')

	models = [graphSage, classification]

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				param.requires_grad = False
				params.append(param)

	embs = graphSage(val_nodes)
	logists = classification(embs)
	_, predicts = torch.max(logists, 1)
	labels_val = labels[val_nodes]
	assert len(labels_val) == len(predicts)
	comps = zip(labels_val, predicts.data)

	print("Validation F1:", f1_score(labels_val, predicts.cpu().data, average="micro"))

	embs = graphSage(test_nodes)
	logists = classification(embs)
	_, predicts = torch.max(logists, 1)
	labels_test = labels[test_nodes]
	assert len(labels_test) == len(predicts)
	comps = zip(labels_test, predicts.data)

	print("Test F1:", f1_score(labels_test, predicts.cpu().data, average="micro"))

	for param in params:
		param.requires_grad = True

def train_classification(dataCenter, graphSage, classification, ds, device, epochs=800):
	print('Training Classification ...')
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	b_sz = 50
	for epoch in range(epochs):
		train_nodes = getattr(dataCenter, ds+'_train')
		labels = getattr(dataCenter, ds+'_labels')
		train_nodes = shuffle(train_nodes)
		batches = math.ceil(len(train_nodes) / b_sz)
		visited_nodes = set()
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
			visited_nodes |= set(nodes_batch)
			labels_batch = labels[nodes_batch]
			embs_batch = graphSage(nodes_batch)

			logists = classification(embs_batch)
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss /= len(nodes_batch)
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()
			
			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			c_optimizer.step()
			c_optimizer.zero_grad()

		evaluate(dataCenter, ds, graphSage, classification, device)
	return classification

def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, num_neg, device, learn_method):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')

	train_nodes = shuffle(train_nodes)

	models = [graphSage, classification]
	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)

	optimizer = torch.optim.SGD(params, lr=0.7)
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()

	batches = math.ceil(len(train_nodes) / b_sz)

	visited_nodes = set()
	for index in range(batches):
		nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning
		nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
		visited_nodes |= set(nodes_batch)

		# get ground-truth for the nodes batch
		labels_batch = labels[nodes_batch]

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		embs_batch = graphSage(nodes_batch)

		if learn_method == 'sup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			loss = loss_sup
		elif learn_method == 'plus_unsup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			# unsuperivsed learning
			loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_sup + loss_net
		else:
			loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
			loss = loss_net

		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

		# if learn_method == 'unsup':
		# 	# train classification, detached from the current graph
		# 	classification.init_params()
		# 	for k in range(100):
		# 		logists = classification(embs_batch.detach())
		# 		loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
		# 		loss_sup /= len(nodes_batch)
		# 		loss_sup.backward()
		# 		nn.utils.clip_grad_norm_(classification.parameters(), 5)
		# 		c_optimizer.step()
		# 		c_optimizer.zero_grad()

		#if visited_nodes == set(train_nodes):
	return graphSage, classification


# def run_cora(device, dataCenter, data):
# 	feat_data, labels, adj_lists = data
# 	test_indexs, val_indexs, train_indexs = split_data(feat_data.size(0))

# 	features = torch.FloatTensor(feat_data).to(device)
# 	print(feat_data.size())

	# agg_enc1 = Aggregator_Encoder(features, features.size(0), dataCenter.config['setting.hidden_emb_size'], adj_lists)
	# agg_enc1.to(device)
	# h1 = agg_enc1(nodes)
	# agg2 = MeanAggregator(lambda nodes : enc1(nodes).t())
	# enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2, base_model=enc1)

# 	enc1.num_samples = 5
# 	enc2.num_samples = 5

# 	graphsage = SupervisedGraphSage(7, enc2)
# #	graphsage.cuda()
# 	rand_indices = np.random.permutation(num_nodes)
# 	test = rand_indices[:1000]
# 	val = rand_indices[1000:1500]
# 	train = list(rand_indices[1500:])

# 	optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
# 	times = []
# 	for batch in range(100):
# 		batch_nodes = train[:256]
# 		random.shuffle(train)
# 		start_time = time.time()
# 		optimizer.zero_grad()
# 		loss = graphsage.loss(batch_nodes, 
# 				Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
# 		loss.backward()
# 		optimizer.step()
# 		end_time = time.time()
# 		times.append(end_time-start_time)
# 		print batch, loss.data[0]

# 	val_output = graphsage.forward(val) 
# 	print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
# 	print "Average batch time:", np.mean(times)