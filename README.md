Eyetracking
===========

A thing for reading raw videos from the Avotec eyetracker.

Sample usage that I literally copy-pasted from a notebook:

```
calibStart = (1, 26, 21, 614)
dataStart = (1, 14, 33, 395)
calibrator = EyetrackingCalibrator.EyetrackingCalibrator('./20181010TZ_et/eyecalib_endof_run01.avi',
                                   calibStart, calibrationDelay = 0, calibrationDuration = 2.1,
                                   calibrationPositions = EyetrackingCalibrator.EyetrackingCalibrator.GeneratePoints(DPIUnscaleFactor = 1.407394),
                                   calibrationOrder = EyetrackingCalibrator.EyetrackingCalibrator.CalibrationOrder35,
                                   templates = True)
calibrator.FindPupils((100, 220, 75, 175), blur = 7, minRadius = 15, maxRadius = 23, windowSize = 29)
calibrator.EstimateCalibrationPointPositions()
calibrator.Fit()
dataPupilFinder = TemplatePupilFinder.TemplatePupilFinder('/E/New Driving Data/20181010TZ_et/driving_run0{}.avi'.format(i + 1))
dataPupilFinder.window = (100, 220, 75, 175)
dataPupilFinder.FindPupils()
dataPupilFinder.FilterPupils(7)
frameFiles = glob('./20181010TZ/Frames/Run 01/*')
frameFiles.sort()
frames = []
for frame in range(len(frameFiles)):
    frames.append(io.imread(frameFiles[frame]))
frames = numpy.array(frames)
eyetracker = Eyetracker.Eyetracker(calibrator = calibrator, dataPupilFinder = dataPupilFinder,
                                   calibrationStart = calibStart, dataStart = dataStart)
eyetracker.RecenterFrames(frames, './20181010TZ/Frames Recentered/Run 0{}/'.format(i + 1))
```

Calibrates eyetracking, reads in frames, and writes eyetracking recentered frames to a folder