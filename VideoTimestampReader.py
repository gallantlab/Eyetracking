import numpy
from .VideoReader import VideoReader
from zipfile import ZipFile
from .EyetrackingUtilities import SaveNPY, ReadNPY, parallelize

SECONDS_SYMBOL = numpy.array([[0,   0,   0,   0,   0,   0,   0,   0],
							  [0,   0,   0,   0,   0,   0,   0,   0],
							  [0,   0,   0,   0,   0,   0,   0,   0],
							  [0,   0,   0,   0,   0,   0,   0,   0],
							  [0,   0, 255, 255, 255, 255,   0,   0],
							  [0, 255, 255,   0, 0,   255, 255,   0],
							  [0, 255, 255,   0,   0,   0,   0,   0],
							  [0,   0, 255, 255, 255, 255,   0,   0],
							  [0,   0,   0,   0,   0, 255, 255,   0],
							  [0, 255, 255,   0,   0, 255, 255,   0],
							  [0,   0, 255, 255, 255, 255,   0,   0],
							  [0,   0,   0,   0,   0,   0,   0,   0]]).ravel()
"""
Hard-coded array for the shape of the seconds symbol burned into the video
"""

class VideoTimestampReader(VideoReader):
	"""
	Gets video frame timestamps from raw eyetracking videos.
	See eyetracker_timestamps
	"""
	templates = numpy.load('/D/Repositories/Eyetracking/digit-templates.npy')
	"""
	@cvar: number templates for reading timestamps
	@type: numpy.ndarray
	@todo: change this to be not hard-coded for thalidomide
	"""
	flats = []
	for i in range(10):
		flats.append(templates[i, :, :].ravel())
	numberTemplates = numpy.stack(flats)
	"""
	@cvar: flattened array of the number templates
	@type: numpy.ndarray
	"""

	@staticmethod
	def GetTimeStampForFrames(frames):
		"""
		Parallelizable function for getting timestamps for a bunch of frames
		@param frames: 	[frame, w, h, 3] frames array
		@return:
		"""

		# separate data in memory because that way the processes won't all have to read from the
		# same memory and deal with concurrency slowdowns
		templates = numpy.load('/D/Repositories/Eyetracking/digit-templates.npy')
		flats = []
		for i in range(10):
			flats.append(templates[i, :, :].ravel())
		numberTemplates = numpy.stack(flats)
		dat = numpy.zeros([frames.shape[0], 4])

		def matchDigit(image):
			"""
			What number is this image?
			@param image: 	2d numpy array
			@return: int, number
			"""
			corrs = []
			flatImage = image.ravel()
			for i in range(10):
				corrs.append(numpy.corrcoef(flatImage, numberTemplates[i, :])[0, 1])
			return numpy.argmax(corrs)

		frames[frames < 255] = 0
		for frameIndex in range(frames.shape[0]):
			frame = frames[frameIndex, :, :]	# red channel
	
			hours = int(matchDigit(frame[:, 7:15]) * 10 + matchDigit(frame[:, 15:23]))		# eyetracker_timestamps.im2hrs()
			minutes = int (matchDigit(frame[:, 35:43]) * 10 + matchDigit(frame[:, 43:51]))	# eyetracker_timestamps.im2mins()
	
			# eyetracker_timestamps.im2seconds() and eyetracker_timestamps.im2secs()
			c = frame[:, 103:111]
			if numpy.corrcoef(c.ravel(), SECONDS_SYMBOL)[0, 1] > 0.99:
				# shift left for miliseconds
				a, b, c = (frame[:, 87 - 8:95 - 8], frame[:, 95 - 8:103 - 8], frame[:, 103 - 8:111 - 8])
	
				# only one leading left digit
				seconds = matchDigit(frame[:, 67:75])
	
			else:
				a, b = (frame[:, 87:95], frame[:, 95:103])
	
				seconds = int(matchDigit(frame[:, 67:75]) * 10 + matchDigit(frame[:, 75:83]))
	
			milliseconds = int(matchDigit(a) * 100 + matchDigit(b) * 10 + matchDigit(c))
			dat[frameIndex, :] = [hours, minutes, seconds, milliseconds]
		return dat

	def __init__(self, videoFileName = None, other = None):
		"""
		Constructor
		@param videoFileName:	str?, video file
		@param other:			VideoReader?, other object to init from
		"""
		super(VideoTimestampReader, self).__init__(videoFileName, other)
		self.numberTemplates = VideoTimestampReader.numberTemplates
		"""
		@ivar: Number templates used for reading timestamps
		@type: numpy.ndarray
		"""

		self.time = numpy.zeros([self.nFrames, 4])  			# [t x 4 (HH MM SS MS)] timestamps on the rawFrames
		"""
		@ivar: Timestamps that have been read out. Colums are Hour, Minute, Second, Milliseconds
		@type: numpy.ndarray<int>
		"""
		# self.frames = self.rawFrames[:, :, :, 2].copy()		# red channel only
		# self.frames[self.frames < 255] = 0					# binarize
		self.isParsed = False
		"""
		@ivar: Have we already parsed the video?
		@type: bool
		"""


	def InitFromOther(self, other):
		"""
		Jank copy constructor
		@param other: VideoTimeStampReader object
		@return:
		"""
		super(VideoTimestampReader, self).InitFromOther(other)
		self.time = other.time.copy()
		self.isParsed = other.isParsed


	@staticmethod
	def MatchDigit(image):
		"""
		What number is this image?
		@param image: 	2d numpy array
		@return: int, number
		"""
		corrs = []
		flatImage = image.ravel()
		for i in range(10):
			corrs.append(numpy.corrcoef(flatImage, VideoTimestampReader.numberTemplates[i, :])[0, 1])
		return numpy.argmax(corrs)


	def GetTimeStampForFrame(self, frameIndex):
		"""
		Gets the timestamp on a single frame. See eyetracker_timestamps.image2time()
		@param frameIndex: 	int, frame to parse
		@return:
		"""
		if self.frame is None:
			self.frame = numpy.zeros([self.height, self.width])

		numpy.copyto(self.frame, self.rawFrames[frameIndex, :, :, 2])	# red channel
		self.frame[self.frame < 255] = 0

		hours = int(VideoTimestampReader.MatchDigit(self.frame[195:207, 7:15]) * 10 + VideoTimestampReader.MatchDigit(self.frame[195:207, 15:23]))		# eyetracker_timestamps.im2hrs()
		minutes = int (VideoTimestampReader.MatchDigit(self.frame[195:207, 35:43]) * 10 + VideoTimestampReader.MatchDigit(self.frame[195:207, 43:51]))	# eyetracker_timestamps.im2mins()

		# eyetracker_timestamps.im2seconds() and eyetracker_timestamps.im2secs()
		c = self.frame[195:207, 103:111]
		if numpy.corrcoef(c.ravel(), SECONDS_SYMBOL)[0, 1] > 0.99:
			# shift left for miliseconds
			a, b, c = (self.frame[195:207, 87 - 8:95 - 8], self.frame[195:207, 95 - 8:103 - 8], self.frame[195:207, 103 - 8:111 - 8])

			# only one leading left digit
			seconds = VideoTimestampReader.MatchDigit(self.frame[195:207, 67:75])

		else:
			a, b = (self.frame[195:207, 87:95], self.frame[195:207, 95:103])

			seconds = int(VideoTimestampReader.MatchDigit(self.frame[195:207, 67:75]) * 10 + VideoTimestampReader.MatchDigit(self.frame[195:207, 75:83]))

		milliseconds = int(VideoTimestampReader.MatchDigit(a) * 100 + VideoTimestampReader.MatchDigit(b) * 10 + VideoTimestampReader.MatchDigit(c))

		self.time[frameIndex, :] = [hours, minutes, seconds, milliseconds]


	def ParseTimestamps(self, nThreads = 1):
		"""
		Parses timestamps from the images
		@param nThreads:	int, number of threads to sue
		@return:
		"""
		self.frame = None
		### === parallel for ===
		if nThreads == 1:
			for frame in range(self.nFrames):
				self.GetTimeStampForFrame(frame)
		else:
			chunkSize = int(self.nFrames / nThreads)
			frameChunks = []
			for i in range(nThreads):
				start = chunkSize * i
				end = start + chunkSize
				if (i == (nThreads - 1)):
					end = self.nFrames
				frameChunks.append(self.rawFrames[start:end, 195:207, :125, 2].copy())
			results = parallelize(VideoTimestampReader.GetTimeStampForFrames, frameChunks, nThreads)
			self.time = numpy.vstack(tuple(results))
		self.isParsed = True


	def FindOnsetFrame(self, H, M, S, MS, returnDiff = False):
		"""
		Find the frame closest to a given time
		@param H: 			int, hour
		@param M: 			int, minute
		@param S: 			int, seconds
		@param MS: 			int, milliseconds
		@param returnDiff:	bool, return also the difference from the desired times on this frame?
		@return: closest frame, int, and time difference between that frame and this time, in ms, int
		"""
		if not self.isParsed:
			self.ParseTimestamps()

		time = self.time[:, 0] * 3600000 + self.time[:, 1] * 60000 + self.time[:, 2] * 1000 + self.time[:, 3]	# in ms
		desiredTime = H * 3600000 + M * 60000 + S * 1000 + MS

		diffFromDesired = time - desiredTime
		frame = numpy.argmin(numpy.abs(diffFromDesired))
		diff = diffFromDesired[frame]

		if returnDiff:
			return frame, diff
		else:
			return frame

	def Save(self, fileName = None, outFile = None):
		"""
		Save out information
		@param fileName: 	str?, name of file to save, must be not none if fileObject is None
		@param outFile: 	zipfile?, existing object to write to
		@return:
		"""
		closeOnFinish = outFile is None  # we close the file only if this is the actual function that started the file

		if outFile is None:
			outFile = ZipFile(fileName, 'w')

		super(VideoTimestampReader, self).Save(None, outFile)

		SaveNPY(self.time, outFile, 'time.npy')

		if closeOnFinish:
			outFile.close()

	def Load(self, fileName = None, inFile = None):
		"""
		Loads in information
		@param fileName: 	str? name of file to read, must not be none if infile is none
		@param inFile:		zipfile? existing object to read from
		@return:
		"""
		closeOnFinish = inFile is None
		if inFile is None:
			inFile = ZipFile(fileName, 'r')

		super(VideoTimestampReader, self).Load(None, inFile)
		self.time = ReadNPY(inFile, 'time.npy')
		self.isParsed = True

		if closeOnFinish:
			inFile.close()