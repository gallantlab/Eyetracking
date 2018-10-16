import numpy
import cv2
from EyetrackingUtilities import map

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

class VideoTimestampReader:
	"""
	Gets video frame timestamps from raw eyetracking videos
	"""

	def __init__(self, videoFileName, BGR = None):
		"""
		Constructor
		@param videoFileName:	str, video file
		"""
		self.fileName = videoFileName
		self.video = cv2.VideoCapture(videoFileName)
		self.fps = 0
		self.width = 0
		self.height = 0
		self.duration = 0		# in seconds
		self.nFrames = 0
		templates = numpy.load('./digit-templates.npy')
		flats = []
		for i in range(10):
			flats.append(templates[i, :, :].ravel())
		self.numberTemplates = numpy.stack(flats)

		self.GetVideoInfo()
		self.time = numpy.zeros([self.nFrames, 4])  # [t x 3 (HH MM SS MS)] timestamps on the frames
		self.frames = None		# [t x w x h] video frames; the timestamps are in red, so we keep only that channel
		self.LoadFrames()


	def LoadFrames(self):
		"""
		Loads frames to memory
		@return:
		"""
		frames = []
		success, frame = self.video.read()
		redIndex = 2 # BGR encoding in the videos
		while success:
			frames.append(frame[:, :, redIndex])
			success, frame = self.video.read()
		self.frames = numpy.stack(frames)
		self.frames[self.frames != 255] = 0		# binarize image


	def GetVideoInfo(self):
		"""
		Gets video info
		@return:
		"""
		self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = self.video.get(cv2.CAP_PROP_FPS)
		self.nFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
		self.duration = self.nFrames / self.fps  # duration in seconds


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

		frame = self.frames[frameIndex, :, :]

		hours = int(self.MatchDigit(frame[195:207, 7:15]) * 10 + self.MatchDigit(frame[195:207, 15:23]))		# eyetracker_timestamps.im2hrs()
		minutes = int (self.MatchDigit(frame[195:207, 35:43]) * 10 + self.MatchDigit(frame[195:207, 43:51]))	# eyetracker_timestamps.im2mins()

		# eyetracker_timestamps.im2seconds() and eyetracker_timestamps.im2secs()
		c = frame[195:207, 103:111]
		if numpy.corrcoef(c.ravel(), SECONDS_SYMBOL)[0, 1] > 0.99:
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
		for frame in range(self.nFrames):
			self.GetTimeStampForFrame(frame)