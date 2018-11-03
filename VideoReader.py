import numpy
import cv2

class VideoReader(object):
	"""
	Base class that loads video rawFrames to an array
	"""
	def __init__(self, videoFileName = None, other = None):
		"""
		Constructor
		@param videoFileName:	str?, video file
		@param other:			VideoReader?, use for copy constructing
		"""
		self.fileName = videoFileName

		self.fps = 0
		self.width = 0
		self.height = 0
		self.duration = 0		# in seconds
		self.nFrames = 0

		self.rawFrames = None		# [t x w x h x 3] BGR video rawFrames
		self.frames = None			# [t x w x h] grayscale frames
		self.frame = None

		self.video = None			# object for reading files

		self.isVidCap = None		# bool?

		if (self.fileName):
			self.GetVideoInfo()
			self.LoadFrames()
		else:
			self.InitFromOther(other)


	def InitFromOther(self, other):
		"""
		A jank copy constructor
		@param other: 	VideoReader object
		@return:
		"""
		self.rawFrames = other.rawFrames
		self.fps = other.fps
		self.width = other.width
		self.height = other.height
		self.duration = other.duration
		self.nFrames = other.nFrames


	def LoadFrames(self):
		"""
		Loads rawFrames to memory
		@return:
		"""
		if (self.isVidCap is None):
			self.GetVideoInfo()

		if self.isVidCap:
			frames = []
			success, frame = self.video.read()
			while success:
				frames.append(frame)
				success, frame = self.video.read()
			self.rawFrames = numpy.stack(frames)
			self.video.release()
		else:
			import subprocess

			bufferSize = self.width * self.height * 3

			command = ['ffmpeg',
					   '-y',
					   '-f', 'avi',
					   '-i', '-',
					   '-f', 'rawvideo',
					   '-pix_fmt', 'rgb24',
					   '-vcodec', 'rawvideo',
					   '-']

			pipe = subprocess.Popen(command, stdin = subprocess.PIPE, stdout = subprocess.PIPE, bufsize = bufferSize)
			rawFrameData = pipe.communicate(self.video)[0]

			frames = numpy.fromstring(rawFrameData, dtype = numpy.uint8)
			self.rawFrames = frames.reshape(-1, self.width, self.height, 3)
			try:
				pipe.kill()
			except OSError:
				pass
			del pipe
			del self.video

		# self.rawFrames = self.rawFrames[:, :, :, 2]		# only keep red channel since that's where the time info is and everything else is is b/w


	def GetVideoInfo(self):
		"""
		Gets video info
		@return:
		"""
		vc = cv2.VideoCapture(self.fileName)
		if (vc.isOpened()):
			self.video = vc
			self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.fps = self.video.get(cv2.CAP_PROP_FPS)
			self.nFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
			self.duration = self.nFrames / self.fps  # duration in seconds
			self.isVidCap = True
		else:	# not a video file on disk
			self.isVidCap = False
			if self.fileName[:3] == 's3:':		# file is on s3; assumed to the gzipped
				import cottoncandy

				fileName = self.fileName[5:]	# since s3 files being with 's3://'
				bucket = fileName.split('/')[0]
				cloud = cottoncandy.get_interface(bucket)
				zippedData = cloud.download_stream(fileName[(len(bucket) + 1):]).content
			else:								# zipped file on disk
				file = open(self.fileName)
				zippedData = file.read()
				file.close()
			self.video = ''
			zipFile = cottoncandy.utils.GzipInputStream(zippedData, 20 * (2 ** 20))
			while True:
				chunk = zipFile.read(10 * (2 ** 20))
				if not chunk:
					break
				self.video += chunk
			del zippedData, zipFile

			import struct
			metadata = struct.unpack('i' * 14, self.video[32:(32 + 56)])
			self.width = metadata[8]
			self.height = metadata[9]
			self.nFrames = metadata[4]
			self.fps = int(1 / (metadata[0] * 1000000))
			self.duration = metadata[0] * metadata[4] / 1000000.0


	def WriteVideo(self, outFileName, frames = None):
		"""
		Writes a video out to disk
		@param outFileName:
		@param frames:
		@return:
		"""

		if frames is None:
			frames = self.frames

