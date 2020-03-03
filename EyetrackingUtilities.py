import numpy
import cPickle
import re
import io

import multiprocessing

def parallelize(function, iterable, nThreads = multiprocessing.cpu_count()):
	"""
	Parallelizes a function. Copied from pycortex so as to not have that import
	@param function:
	@param iterable:
	@param nThreads:
	@return:
	"""
	inputQueue = multiprocessing.Queue()
	outputQueue = multiprocessing.Queue()
	length = multiprocessing.Value('i', 0)

	def _fill(iterable, nThreads, inputQueue, outputQueue):
		for data in enumerate(iterable):
			inputQueue.put(data)
			length.value += 1
		for _ in range(nThreads * 2):
			inputQueue.put((-1, -1))

	def _func(proc, inputQueue, outputQueue):
		index, data = inputQueue.get()
		while index != -1:
			outputQueue.put((index, function(data)))
			index, data = inputQueue.get()

	filler = multiprocessing.Process(target = _fill, args = (iterable, nThreads, inputQueue, outputQueue))
	filler.daemon = True
	filler.start()
	for i in range(nThreads):
		proc = multiprocessing.Process(target = _func, args = (i, inputQueue, outputQueue))
		proc.daemon = True
		proc.start()

	try:
		iterlen = len(iterable)
	except:
		filler.join()
		iterlen = length.value

	data = [[]] * iterlen
	for _ in range(iterlen):
		index, result = outputQueue.get()
		data[index] = result

	return data


def TimeToSeconds(time):
	"""
	Converts a timestamp to just seconds elapsed
	@param time: 	tuple<int, int, int, int> of HH:MM:SS.SSS timestamp
	@return: 	float, seconds equivalence
	"""
	return 3600 * time[0] + 60 * time[1] + time[2] + 0.001 * time[3]


def ParseHistoryForStartTTLs(historyFileName, useMovieMarkers = True, TR = 2.0, onset = False, threshold = 1.5):
	"""
	Parses the history file from Avotec for the start TTL timings for runs
	@param historyFileName:	str, name of history file from avotec
	@param useMovieMarkers:	bool, use the start/stop save movie entries to calculate runs? if true, the TR, onset, and threshold arguments are useless
	@param TR:				float, TR length used
	@param onset:			bool, use the TTL pulse HI instead of the LO value?
	@param threshold:		float, multiple of the TR interval to use as a threshold as a break?
	@return:	list<tuple<tuple<float>, int>>, first value is the timestamp of the first TTL in a run, and the second is number of TRs in each run
	"""

	historyFile = open(historyFileName, 'r')
	TTLtoken = 'HI' if onset else 'LO'
	TTLs = []
	starts = []
	lastTime = (0, 0, 0, 0)
	duplicates = 0

	if useMovieMarkers:
		nTTLs = 0
		isStarted = False
		start = None
		line = historyFile.readline()
		while line != '':
			tokens = line.split()
			if len(tokens) > 0:
				if tokens[-1] == 'saveMovie[0]:':
					isStarted = True
					start = None
					nTTLs = 0
				elif tokens[3] == 'Closing':
					isStarted = False
					starts.append((start, nTTLs))
				if isStarted:
					if tokens[-1] == TTLtoken and tokens[4] == 'TTL':
						time = tuple([int(token) for token in re.split('[:\.]', tokens[0])])
						if start is None:
							start = time
						if ((TimeToSeconds(time) - TimeToSeconds(lastTime)) > 0.1):	# long enough of an interval since last one such that it's not a duplicate
							nTTLs += 1
							lastTime = time
						else:
							duplicates += 1
			line = historyFile.readline()

	else:
		line = historyFile.readline()
		while line != '':
			tokens = line.split()
			if len(tokens) > 0 and tokens[-1] == TTLtoken:
				time = tuple([int(token) for token in re.split('[:\.]', tokens[0])])
				if (TimeToSeconds(time) - TimeToSeconds(lastTime) > 0.1):  # long enough of an interval since last one such that it's not a duplicate
					TTLs.append(time)
					lastTime = time
				else:
					duplicates += 1
			line = historyFile.readline()

		firstInRun = TTLs[0]
		nTRs = 1
		for i in range(1, len(TTLs) - 1):
			this = TTLs[i]
			last = TTLs[i - 1]
			dt = TimeToSeconds(this) - TimeToSeconds(last)
			if dt > threshold * TR:
				starts.append((firstInRun, nTRs))
				firstInRun = this
				nTRs = 1
			else:
				nTRs += 1
		starts.append((firstInRun, nTRs + 1))	# account for last run in which there's no faraway TR to make the math work

	historyFile.close()
	print('{} duplicated TTLs'.format(duplicates))
	return starts


def SaveNPY(array, zipfile, name):
	"""
	Saves a numpy array into a zip
	@param array: 	numpy array
	@param zipfile: ZipFile
	@param name: 	str, name to save
	@return:
	"""
	arrayFile = io.BytesIO()
	numpy.save(arrayFile, array)
	arrayFile.seek(0)
	zipfile.writestr(name, arrayFile.read())
	arrayFile.close()
	del arrayFile


def ReadNPY(zipfile, subFileName):
	"""
	Reads a saved npy from inside the zip
	@param zipfile: 		ZipFile
	@param subFileName: 	str, file name
	@return: array
	"""
	arrayFile = io.BytesIO(zipfile.read(subFileName))
	arrayFile.seek(0)
	out = numpy.load(arrayFile)
	del arrayFile
	return out


def ReadPickle(zipfile, subFileName):
	"""
	Reads a saved pickle from inside the zip
	@param zipfile: 		ZipFile
	@param subFileName: 	str, file name
	@return: object
	"""
	objFile = io.BytesIO(zipfile.read(subFileName))
	objFile.seek(0)
	out = cPickle.load(objFile)
	del objFile
	return out


def SavePickle(obj, zipfile, name):
	"""
	Pickles an object into a zip
	@param obj: 	object
	@param zipfile: ZipFile
	@param name: 	str, name to save
	@return:
	"""
	pickleFile = io.BytesIO()
	cPickle.dump(obj, pickleFile)
	pickleFile.seek(0)
	zipfile.writestr(name, pickleFile.read())
	pickleFile.close()
	del pickleFile