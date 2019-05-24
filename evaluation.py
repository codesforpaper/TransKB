import numpy as np
import tensorflow as tf
from scipy.stats import rankdata

class Evaluation(object):
	"""docstring for Evaluation"""
	def __init__(self, args, train, valid, test, entity_array, entity2id, Trans, sess):
		self.args = args
		self.entity_array = entity_array
		self.entity2id = entity2id
		self.Trans = Trans
		self.sess = sess
		self.train = train
		self.valid = valid
		self.test = test 

	def fixedPositionEmbedding(self, batchSize, sequenceLen):
		embeddedPosition = []
		for batch in range(batchSize):
			x = []
			for step in range(sequenceLen):
				a = np.zeros(sequenceLen)
				a[step] = 1
				x.append(a)
			embeddedPosition.append(x)
		return np.array(embeddedPosition, dtype="float32")

	def _positionEmbedding(self, batchSize, sequenceLen, embeddingSize, scope="positionEmbedding"):
		# 生成可训练的位置向量
		# batchSize = self.config.batchSize
		# sequenceLen = self.config.sequenceLength
		# embeddingSize = self.config.model.embedding_dim

		# 生成位置的索引，并扩张到batch中所有的样本上
		positionIndex = np.tile(np.expand_dims(range(sequenceLen), 0), [batchSize, 1])

		# 根据正弦和余弦函数来获得每个位置上的embedding的第一部分​https://github.com/Kyubyong/transformer/issues/3
		positionEmbedding = np.array([[pos / np.power(10000, (i-i%2) / embeddingSize) for i in range(embeddingSize)] 
									for pos in range(sequenceLen)])

		# 然后根据奇偶性分别用sin和cos函数来包装
		positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
		positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

		# 将positionEmbedding转换成tensor的格式
		positionEmbedding_ = np.array(positionEmbedding, dtype='float32')

		# 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
		# positionEmbedded = np.take(positionEmbedding_, positionIndex)
		positionEmbedded =[]
		for item in positionIndex:
			tmp =[]
			for id in item:
				tmp.append(positionEmbedding_[id])
			positionEmbedded.append(tmp)

		return np.array(positionEmbedded, dtype="float32")

	# Predict function to predict scores for test data
	def predict(self, x_batch, y_batch, writer=None):
		Trans = self.Trans
		sess = self.sess
		batch_size = len(x_batch)
		sequenceLength = self.args.sequenceLength
		feed_dict = {
			Trans.inputX: x_batch,
			Trans.inputY: y_batch,
			Trans.dropoutKeepProb: 1.0,
			# Trans.embeddedPosition: self.fixedPositionEmbedding(batch_size, sequenceLength),
			Trans.embeddedPosition: self._positionEmbedding(batch_size,sequenceLength,self.args.embedding_dim)
		}
		scores = sess.run([Trans.predictions], feed_dict)
		return scores

	def test_prediction(self, x_batch, y_batch, head_or_tail='head'):
		hits10 = 0.0
		mrr = 0.0
		mr = 0.0

		for i in range(len(x_batch)):
			# print("x_batch id :{}".format(i))
			new_x_batch = np.tile(x_batch[i], (len(self.entity2id), 1))
			new_y_batch = np.tile(y_batch[i], (len(self.entity2id), 1))
			if head_or_tail == 'head':
				new_x_batch[:, 0] = self.entity_array
			else:  # 'tail'
				new_x_batch[:, 2] = self.entity_array

			lstIdx = []
			for tmpIdxTriple in range(len(new_x_batch)):
				tmpTriple = (new_x_batch[tmpIdxTriple][0], new_x_batch[tmpIdxTriple][1],
					new_x_batch[tmpIdxTriple][2])
				if (tmpTriple in self.train) or (tmpTriple in self.valid) or (tmpTriple in self.test): #also remove the valid test triple
					lstIdx.append(tmpIdxTriple)
			new_x_batch = np.delete(new_x_batch, lstIdx, axis=0)#全都不属于train,valid,test里的
			new_y_batch = np.delete(new_y_batch, lstIdx, axis=0)

			#thus, insert the valid test triple again, to the beginning of the array
			new_x_batch = np.insert(new_x_batch, 0, x_batch[i], axis=0) #thus, the index of the valid test triple is equal to 0
			new_y_batch = np.insert(new_y_batch, 0, y_batch[i], axis=0)

			# while len(new_x_batch) % ((int(args.neg_ratio) + 1) * args.batch_size) != 0:
			#    new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
			#    new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)

			results = []
			listIndexes = range(0, len(new_x_batch), (int(self.args.neg_ratio) + 1) * self.args.batch_size)
			for tmpIndex in range(len(listIndexes) - 1):
				results = np.append(results, self.predict(
					new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
					new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]]))
			results = np.append(results, self.predict(new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:]))

			results = np.reshape(results, -1)
			results_with_id = rankdata(results, method='ordinal')
			_filter = results_with_id[0]

			mr += _filter
			mrr += 1.0 / _filter
			if _filter <= 10:
				hits10 += 1

		return np.array([mr, mrr, hits10])/len(x_batch)

	# def evaluation(self, x_test, y_test):
	# 	startid = 0
	# 	endid = self.args.batch_size
	# 	head_results, tail_results = [0,0,0],[0,0,0]
	# 	while endid <= len(x_test):
	# 		print(endid)
	# 		head_results += self.test_prediction(
	# 			x_test[startid:endid],
	# 			y_test[startid:endid],
	# 			head_or_tail='head'
	# 		)
	# 		tail_results += self.test_prediction(
	# 			x_test[startid:endid],
	# 			y_test[startid:endid],
	# 			head_or_tail='tail'	
	# 		)
	# 		startid = endid
	# 		endid = endid + self.args.batch_size
	# 	if endid > len(x_test):
	# 		endid = len(x_test)
	# 		head_results += self.test_prediction(
	# 			x_test[startid:endid],
	# 			y_test[startid:endid],
	# 			head_or_tail='head'
	# 		)
	# 		tail_results += self.test_prediction(
	# 			x_test[startid:endid],
	# 			y_test[startid:endid],
	# 			head_or_tail='tail'
	# 		)

	# 	return (head_results+tail_results) / (2 * len(x_test))
	def evaluation(self):
		x_test = np.array(list(self.test.keys())).astype(np.int32)
		y_test = np.array(list(self.test.values())).astype(np.float32)
		len_test = len(x_test)
		batch_test = int(len_test / (self.args.num_splits - 1))
		# print(batch_test)
		if self.args.testIdx < (self.args.num_splits - 1):
			head_results = self.test_prediction(
				x_test[batch_test * self.args.testIdx: batch_test * (self.args.testIdx + 1)],
				y_test[batch_test * self.args.testIdx: batch_test * (self.args.testIdx + 1)],
				head_or_tail='head')
			tail_results = self.test_prediction(
				x_test[batch_test * self.args.testIdx: batch_test * (self.args.testIdx + 1)],
				y_test[batch_test * self.args.testIdx: batch_test * (self.args.testIdx + 1)],
				head_or_tail='tail')
		else:
			head_results = self.test_prediction(x_test[batch_test * self.args.testIdx: len_test],
				y_test[batch_test * self.args.testIdx: len_test],head_or_tail='head')
			tail_results = self.test_prediction(x_test[batch_test * self.args.testIdx: len_test],
				y_test[batch_test * self.args.testIdx: len_test],head_or_tail='tail')
		return head_results,tail_results

	# head_results = test_prediction(
        # x_test[batch_test * args.testIdx: batch_test * (args.testIdx + 1)],
        # y_test[batch_test * args.testIdx: batch_test * (args.testIdx + 1)],
        # head_or_tail='head')
    # tail_results = test_prediction(
    #     x_test[batch_test * args.testIdx: batch_test * (args.testIdx + 1)],
    #     y_test[batch_test * args.testIdx: batch_test * (args.testIdx + 1)],
    #     head_or_tail='tail')
    # return (head_results+tail_results) / (2 * len_test)
