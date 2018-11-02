import numpy
from VideoReader import VideoReader

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

class VideoTimestampReader(VideoReader):
	"""
	Gets video frame timestamps from raw eyetracking videos.
	See eyetracker_timestamps
	"""

	def __init__(self, videoFileName = None, other = None):
		"""
		Constructor
		@param videoFileName:	str?, video file
		@param other:			VideoReader?, other object to init from
		"""
		super(VideoTimestampReader, self).__init__(videoFileName, other)

		templates = numpy.load('./digit-templates.npy')
		flats = []
		for i in range(10):
			flats.append(templates[i, :, :].ravel())
		self.numberTemplates = numpy.stack(flats)

		self.time = numpy.zeros([self.nFrames, 4])  		# [t x 3 (HH MM SS MS)] timestamps on the rawFrames
		self.frames = self.rawFrames.copy()					# red channel only
		self.frames[self.frames < 255] = 0					# binarize


	def MatchDigit(self, image):
		"""
		What number is this image?
		@param image: 	2d numpy array
		@return: int, number
		"""
		corrs = []
		flatImage = image.ravel()
		for i in range(10):
			corrs.append(numpy.corrcoef(flatImage, self.numberTemplates[i, :])[0, 1])
		return numpy.argmax(corrs)


	def GetTimeStampForFrame(self, frameIndex):
		"""
		Gets the timestamp on a single frame. See eyetracker_timestamps.image2time()
		@param frameIndex: 	int, frame to parse
		@return:
		"""

		frame = self.frames[frameIndex, :, :]	# red channel

		hours = int(self.MatchDigit(frame[195:207, 7:15]) * 10 + self.MatchDigit(frame[195:207, 15:23]))		# eyetracker_timestamps.im2hrs()
		minutes = int (self.MatchDigit(frame[195:207, 35:43]) * 10 + self.MatchDigit(frame[195:207, 43:51]))	# eyetracker_timestamps.im2mins()

		# eyetracker_timestamps.im2seconds() and eyetracker_timestamps.im2secs()
		c = frame[195:207, 103:111]
		if numpy.corrcoef(c.ravel(), SECONDS_SYMBOL)[0, 1] > 0.95:
			# shift left for miliseconds
			a, b, c = (frame[195:207, 87 - 8:95 - 8], frame[195:207, 95 - 8:103 - 8], frame[195:207, 103 - 8:111 - 8])

			# only one leading left digit
			seconds = self.MatchDigit(frame[195:207, 67:75])

		else:
			a, b = (frame[195:207, 87:95], frame[195:207, 95:103])

			seconds = int(self.MatchDigit(frame[195:207, 67:75]) * 10 + self.MatchDigit(frame[195:207, 75:83]))

		milliseconds = int(self.MatchDigit(a) * 100 + self.MatchDigit(b) * 10 + self.MatchDigit(c))

		self.time[frameIndex, :] = [hours, minutes, seconds, milliseconds]


	def ParseTimestamps(self):
		"""
		Parses timestamps from the images
		@return:
		"""
		### === parallel for ===
		for frame in range(self.nFrames):
			self.GetTimeStampForFrame(frame)


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

		time = self.time[:, 0] * 3600000 + self.time[:, 1] * 60000 + self.time[:, 2] * 1000 + self.time[:, 3]	# in ms
		desiredTime = H * 3600000 + M * 60000 + S * 1000 + MS

		diffFromDesired = time - desiredTime
		frame = numpy.argmin(numpy.abs(diffFromDesired))
		diff = diffFromDesired[frame]

		if returnDiff:
			return frame, diff
		else:
			return frame