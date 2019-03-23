import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(torch.mm(embeds,self.weight), 1)
		return logists

class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""
	def __init__(self, adj_lists, train_nodes):
		super(UnsupervisedLoss, self).__init__()
		self.N_WALKS = 2
		self.WALK_LEN = 3
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes

		self.positive_pairs = []
		self.negtive_pairs = []
		self.unique_nodes_batch = []

	def forward(self):
		pass

	def extend_nodes(self, nodes):
		self.get_positive_nodes(nodes)
		# print(self.positive_pairs)
		self.get_negtive_nodes(nodes)
		# print(self.negtive_pairs)
		self.unique_nodes_batch = set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x])
		return self.unique_nodes_batch

	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	def get_negtive_nodes(self, nodes):
		self.negtive_pairs = []
		for node in nodes:
			neighbors = set([node])
			frontier = set([node])
			for i in range(self.WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
			far_nodes = set(self.train_nodes) - neighbors
			neg_samples = random.sample(far_nodes, self.N_WALKS*self.WALK_LEN) if self.N_WALKS*self.WALK_LEN < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
		return self.negtive_pairs

	def _run_random_walks(self, nodes):
		self.positive_pairs = []
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					# self co-occurrences are useless
					if curr_node != node and curr_node in self.train_nodes:
						self.positive_pairs.append((node,curr_node))
					curr_node = next_node
		return self.positive_pairs
		

class SageLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size, gcn=False): 
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size


		self.gcn = gcn
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, self_feats, aggregate_feats, neighs=None):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		if not self.gcn:
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			combined = aggregate_feats
		combined = F.relu(self.weight.mm(combined.t())).t()
		return combined

class GraphSage(nn.Module):
	"""docstring for GraphSage"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
		super(GraphSage, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.gcn = gcn
		self.device = device
		self.agg_func = agg_func

		self.raw_features = raw_features
		self.adj_lists = adj_lists

		for index in range(1, num_layers+1):
			layer_size = out_size if index != 1 else input_size
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

	def forward(self, nodes_batch):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
		lower_layer_nodes = list(nodes_batch)
		nodes_batch_layers = [(lower_layer_nodes,)]
		for i in range(self.num_layers):
			lower_samp_neighs, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes)
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs))

		assert len(nodes_batch_layers) == self.num_layers + 1

		pre_hidden_embs = self.raw_features
		for index in range(1, self.num_layers+1):
			nb = nodes_batch_layers[index][0]
			pre_neighs = nodes_batch_layers[index-1]
			aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
			sage_layer = getattr(self, 'sage_layer'+str(index))
			if index > 1:
				nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)
			pre_hidden_embs = cur_hidden_embs

		return pre_hidden_embs

	def _nodes_map(self, nodes, hidden_embs, neighs):
		layer_nodes, samp_neighs = neighs
		assert len(samp_neighs) == len(nodes)
		index = [layer_nodes.index(x) for x in nodes]
		return index


	def _get_unique_neighs_list(self, nodes, num_sample=10):
		_set = set
		to_neighs = [self.adj_lists[int(node)] for node in nodes]
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs
		samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		return samp_neighs, unique_nodes_list

	def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
		# samp_neighs, unique_nodes_list = self._get_unique_neighs_list(nodes, num_sample)
		# unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
		# embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]

		unique_nodes_list, samp_neighs = pre_neighs
		unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

		# aggregate the neighbors, discard self nodes
		assert len(nodes) == len(samp_neighs)
		indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
		assert (False not in indicator)
		if not self.gcn:
			samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]

		if len(pre_hidden_embs) == len(unique_nodes_list):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]

		mask = torch.zeros(len(samp_neighs), len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1

		num_neigh = mask.sum(1, keepdim=True)
		# print(mask)
		indexs = [x.nonzero() for x in mask==1]
		aggregate_feats = []
		for feat in [embed_matrix[x.squeeze()] for x in indexs]:
			if len(feat.size()) == 1:
				aggregate_feats.append(feat.view(1, -1))
			else:
				if self.agg_func == 'MEAN':
					aggregate_feats.append(torch.mean(feat,0).view(1, -1))
				elif self.agg_func == 'MAX':
					aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
		aggregate_feats = torch.cat(aggregate_feats, 0)

		# mask = mask.div(num_neigh).to(self.device)
		# aggregate_feats = mask.mm(embed_matrix)
		# print(aggregate_feats.size())
		# assert len(outputs) == len(aggregate_feats)
		# for i, output in enumerate(outputs):
		# 	print(aggregate_feats[i] - output)
		return aggregate_feats
