#!/usr/bin/env python3

from os.path import abspath, basename, splitext
from subprocess import run
import datetime
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import stats

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

class Video:
	
	def __init__(self, videoFile, encoder):
		self.videoFile = videoFile
		self.outputFile = basename(videoFile)
		self.listOfShots = []
		self.iteration = 0
		self.encoder = encoder[0]
		self.minQP = encoder[1]
		self.maxQP = encoder[2]
		

	def generateShotList(self, threshold, guess):
		print("Generating list of shots with a threshold: " + str(threshold))
		shots = self.findShots(self.videoFile, threshold)
		for shot in shots:
			self.listOfShots.append(Shot(shot, guess))


	def findShots(self, video_path, threshold):
	    # Create our video & scene managers, then add the detector.
	    video_manager = VideoManager([video_path])
	    scene_manager = SceneManager()
	    scene_manager.add_detector(ContentDetector(threshold=threshold))

	    # Base timestamp at frame 0 (required to obtain the scene list).
	    base_timecode = video_manager.get_base_timecode()

	    # Improve processing speed by downscaling before processing.
	    video_manager.set_downscale_factor()

	    # Start the video manager and perform the scene detection.
	    video_manager.start()
	    scene_manager.detect_scenes(frame_source=video_manager)

	    # Each returned scene is a tuple of the (start, end) timecode.
	    return scene_manager.get_scene_list(base_timecode)


	def sortShots(self):
		self.listOfShots = sorted(self.listOfShots, key=lambda shot: shot.firstFrame)
		
		
	def transcode(self):
		print("Transcoding ...")
		if self.encoder == "x264":
			self.transcodeX264()
		elif self.encoder == "x265":
			self.transcodeX265()
		else:
			print("Unknown encoder, again!")
			
			
	def finalTranscode(self):
		print("Transcoding ...")
		if self.encoder == "x264":
			self.transcodeX264()
		elif self.encoder == "x265":
			self.transcodeX265()
		else:
			print("Unknown encoder, again!")
			
	
	def transcodeX264(self):
		# Generate I-Frame and Zones strings for FFmpeg
		IFramesString = ""
		zonesString = "zones="
		for shot in self.listOfShots:
			qp = shot.nextQP
			IFramesString += str(shot.firstTime)
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame) + ",q=" + str(qp)
			if shot.firstFrame != self.listOfShots[-1].firstFrame:
				IFramesString += ","
				zonesString += "/"
		
		# Generate complete x264 parameter string for FFmpeg		
		paramsString = "scenecut=0:"
		paramsString += zonesString
		
		# Full FFmpeg command
		x264 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-map", "0:0", "-force_key_frames:v", IFramesString, "-c:v", 
				"libx264", "-preset:v", "medium", "-x264-params", paramsString, 
				"-profile:v", "high", "-color_primaries:v", "bt470bg", "-color_trc:v", 
				"bt709", "-colorspace:v", "smpte170m", "-metadata:s:v", "title\=", 
				"-disposition:v", "default", "-map", "0:1", "-c:a:0", "ac3", "-b:a:0", 
				"640k", "-metadata:s:a:0", "title\=", "-disposition:a:0", "default", 
				"-sn", "-metadata:g", "title\=", self.outputFile]
		
		# Run FFmpeg command
		a = run(x264)
			
	
	def transcodeX265(self):
		# Generate Zones strings for FFmpeg
		zonesString = "zones="
		for shot in self.listOfShots:
			qp = shot.nextQP
			zonesString += str(shot.firstFrame) + "," + str(shot.lastFrame) + ",q=" + str(qp)
			if shot.firstFrame != self.listOfShots[-1].firstFrame:
				zonesString += "/"
		
		# Generate complete x265 parameter string for FFmpeg		
		statsFileName = "stats-" + str(self.iteration) + ".csv"
		paramsString = "scenecut=0:csv=" + statsFileName + ":csv-log-level=1:"
		if self.iteration == 0:
			paramsString += "analysis-save=analysis.dat:analysis-save-reuse-level=10:"
		else:
			paramsString += "analysis-load=analysis.dat:analysis-load-reuse-level=10:"
		paramsString += zonesString
		
		# Full FFmpeg command
		x265 = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", "-i", self.videoFile, 
				"-y", "-map", "0:0", "-c:v", "libx265", "-preset:v", "medium", 
				"-x265-params", x265ParamsString, "-color_primaries:v", 
				"bt470bg", "-color_trc:v", "bt709", "-colorspace:v", "smpte170m", 
				"-metadata:s:v", "title\=", "-disposition:v", "default", "-map", "0:1", 
				"-c:a:0", "ac3", "-b:a:0", "640k", "-metadata:s:a:0", "title\=", 
				"-disposition:a:0", "default", "-sn", "-metadata:g", "title\=", self.outputFile]
		
		# Run FFmpeg command
		a = run(x265)
		
	
	def calculateVideoVMAF(self, modelPath):
		print("Calculating quality ...")
		
		# VMAF CSV File Name
		vmafOut = splitext(self.outputFile)[0] + "-vmaf.csv"
		
		# Assemble select, scale and vmaf filter strings		
		scaleStringMain = "[0:v]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[main]; "
		scaleStringRef = "[1:v]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[ref]; "		
		vmafFilterString = "[main][ref]libvmaf=model_path=" + modelPath
		vmafFilterString += ":log_path=" + vmafOut
		vmafFilterString += ":log_fmt=csv"
		
		# Assemble final filter string
		filterString =  scaleStringMain + scaleStringRef + vmafFilterString
		
		# Assemble FFmpeg command	
		vmafCommand = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", 
						"-r", "24", "-i", self.outputFile, 
						"-r", "24", "-i", self.videoFile, 
						"-filter_complex", filterString, "-f", "null", "-"]
		
		# Run FFmpeg command
		a = run(vmafCommand)
		
		# Read in VMAF CSV file
		vmaf_df = pd.read_csv(vmafOut, usecols=['Frame', 'vmaf'])
		
		# Averages & Percentiles
		vmafMean = vmaf_df['vmaf'].mean()
		vmafHMean = stats.hmean(vmaf_df['vmaf'],axis=0)
		vmafMax = vmaf_df['vmaf'].max()
		vmafP75 = np.percentile(vmaf_df['vmaf'], q=75)
		vmafP25 = np.percentile(vmaf_df['vmaf'], q=25)
		vmafMin = vmaf_df['vmaf'].min()
		
		# Print results
		print("VMAF mean = " + str(vmafMean))
		print("VMAF harmonic mean = " + str(vmafHMean))
		print("VMAF maximum = " + str(vmafMax))
		print("VMAF 75th percentile = " + str(vmafP75))
		print("VMAF 25th percentile = " + str(vmafP25))
		print("VMAF minimum = " + str(vmafMin))
		
		
	def calculateVMAF(self, modelPath, subSample):
		print("Calculating quality ...")
		
		# which shots need analysing
		listOfUnoptimisedShots = []
		firstFrame = 0
		lastFrame = 0
		for shot in self.listOfShots:
			if not shot.isOptimised():
				shotLength = shot.lastFrame - shot.firstFrame
				lastFrame = firstFrame + shotLength
				listOfUnoptimisedShots.append((shot, firstFrame, lastFrame))
				firstFrame = lastFrame + 1
		
		# generate selection string
		lastShotFirstFrame = listOfUnoptimisedShots[-1][0].firstFrame
		selectString = "select="
		for i in listOfUnoptimisedShots:
			shot = i[0]
			selectString += "between(n\," + str(shot.firstFrame) + "\,"
			selectString += str(shot.lastFrame) + ")"
			if shot.firstFrame != lastShotFirstFrame:
				selectString += "+"
				
		# VMAF CSV File Name
		vmafOut = splitext(self.outputFile)[0] + "-vmaf.csv"
		
		# Assemble select, scale and vmaf filter strings
		selectStringMain = "[0:v]" + selectString + "[main]; "
		selectStringRef = "[1:v]" + selectString + "[ref]; "		
		scaleStringMain = "[main]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[main]; "
		scaleStringRef = "[ref]scale=1920x1080:flags=bicubic,settb=AVTB,setpts=PTS-STARTPTS[ref]; "		
		vmafFilterString = "[main][ref]libvmaf=model_path=" + modelPath
		vmafFilterString += ":log_path=" + vmafOut
		vmafFilterString += ":log_fmt=csv:n_subsample=" + str(subSample)
		
		# Assemble final filter string
		filterString = selectStringMain + scaleStringMain + selectStringRef + \
						scaleStringRef + vmafFilterString
		
		# Assemble FFmpeg command	
		vmafCommand = [ "ffmpeg", "-hide_banner", "-v", "error", "-stats", 
						"-r", "24", "-i", self.outputFile, 
						"-r", "24", "-i", self.videoFile, 
						"-filter_complex", filterString, "-f", "null", "-"]
		
		# Run FFmpeg command
		a = run(vmafCommand)
		
		# Read in VMAF CSV file
		vmaf_df = pd.read_csv(vmafOut, usecols=['Frame', 'vmaf'])
		
		# Add QP-VMAF pairs to shot data
		for shot in listOfUnoptimisedShots:
			shotVmaf = vmaf_df.loc[(vmaf_df['Frame'] >= shot[1]) & 
									(vmaf_df['Frame'] <= shot[2])]
			# Calculate average VMAF (mean, harmonic, 25th percentile)
			#aveVmaf = shotVmaf['vmaf'].mean()
			#aveVmaf = stats.hmean(shotVmaf['vmaf'],axis=0)
			aveVmaf = np.percentile(shotVmaf['vmaf'], q=25)
			
			# Add QP-VMAF pair to shot object
			shot[0].addQpVmaf(shot[0].nextQP, aveVmaf)
	
	
	def calculateNewSettings(self, targetVMAF):
		for shot in self.listOfShots:
			shot.calculateNewQP4(targetVMAF, self.minQP, self.maxQP)
			
	
	def printSummary(self, targetVMAF):
		shotNumber = 0
		vmaf = 0.0
		totalNumFrames = 0
		for shot in self.listOfShots:
			numFrames = shot.lastFrame - shot.firstFrame
			totalNumFrames += numFrames
			bestRQ = min(shot.qpVmaf, key=lambda rq: abs(rq[1] - targetVMAF))
			vmaf += numFrames*bestRQ[1]

			print("Shot: " + str(shotNumber))
			print("Frames: " + str(shot.firstFrame) + "-" + str(shot.lastFrame))
			print("Best: " + str(bestRQ[0]) + " - " + str(bestRQ[1]))
			print("Next: " + str(shot.nextQP))
			print("History:")
			for i in shot.qpVmaf:
				print(str(i[0]) + " - " + str(i[1]))
			print()
			
			shotNumber += 1
		print("Average VMAF: " + str(vmaf/totalNumFrames))
	
	
	def printStatus(self):
		numOptimised = 0
		for shot in self.listOfShots:
			if shot.isOptimised():
				numOptimised += 1
		print(str(numOptimised) + " optimised shots out of " + str(len(self.listOfShots)))
		
	
	def isOptimised(self):
		optimised = True
		for shot in self.listOfShots:
			if not shot.isOptimised():
				optimised = False
		return optimised
	
	
	def optimise(self, targetVMAF, modelPath, subSample):
		print("Starting optimising for VMAF: " + str(targetVMAF))
		while not self.isOptimised():
			print("Iteration: " + str(self.iteration+1))
			self.transcode()
			
			self.calculateVMAF(modelPath, subSample)
			
			print("Calculating new setting")
			self.calculateNewSettings(targetVMAF)
			
			#self.printSummary(targetVMAF)
			self.printStatus()
			self.iteration += 1
					
		print("Finished")
		

class Shot:
	
	def __init__(self, scene_list, guess):
		if scene_list[0].get_frames() == 0:
			self.firstFrame = scene_list[0].get_frames()
			self.firstTime = scene_list[0].get_seconds()
		else:
			self.firstFrame = scene_list[0].get_frames()+1
			self.firstTime = scene_list[0].get_seconds()+(1.0/scene_list[0].get_framerate())
		self.lastFrame = scene_list[1].get_frames()
		self.lastTime = scene_list[1].get_seconds()
		self.nextQP = guess
		self.qpVmaf = []
	
	
	def calculateNewQP4(self, targetVMAF, minQP, maxQP):
		# Make sure there is some data to process
		numDataPoints = len(self.qpVmaf)
		if numDataPoints == 0:
			print("Error: Should not have called calculateNewQP before first transcode")
			exit()
			
		# If the most recent VMAF is 100.0 add 1/4th of total QP range to QP
		maxQpStep = int(np.ceil((maxQP-minQP)/4.0))
		if [x[1] for x in self.qpVmaf].count(100.0) == numDataPoints:
			if self.nextQP + maxQpStep <= maxQP:
				nextQP = self.nextQP + maxQpStep
			else:
				nextQP = maxQP
			self.nextQP = nextQP
			return 1
		
		# Sort qpVmaf by QP from low to high
		qpVmafSorted = sorted(self.qpVmaf, key=lambda row: row[0])
			
		# Modelling the qp-vmaf curve as y=100 up to an unknown (possibly negative)
		# qp number refered in the code as "t" for transition. After that the curve
		# is modelled as a 4th order polynomial:
		# a(x-t)^4 + b(x-t)^3 + c(x-t)^2 + d(x-t) + e
		
		# Initially
		t = 0
		
		# If there are multiple VMAF=100 entries remove all but the one with
		# the largest QP
		while [x[1] for x in qpVmafSorted].count(100.0) > 1:
			qpVmafSorted.pop(0)
			
		# If vmaf==100 is in the data, store, set "t" as that qp value then remove it
		if [x[1] for x in qpVmafSorted].count(100.0) == 1:
			t = qpVmafSorted[0][0]
			qpVmafSorted.pop(0)
		
		# "t" provides us the minimum QP we can usefully use for a new guess (anything
		# smaller would produce vmaf=100.0)
		if t > minQP:
			minQP = t
		
		# If there are other QP's with VMAF's too large they can provide a better lower
		# bound for minQP
		for rq in qpVmafSorted:
			if rq[1] > targetVMAF and rq[0] > minQP:
				minQP = rq[0]
			
		# The maximum QP can be lowered to the smallest QP in the qpVmaf array so long
		# as the VMAF is smaller than targetVMAF
		for rq in qpVmafSorted:
			if rq[1] < targetVMAF:
				maxQP = rq[0]
				break
		
		# Use minQP and maxQP to construct array of all allowable QPs
		allowableQPs = np.arange(minQP, maxQP+1, 1)
			
		# Separate into x's and y's and recalculate number of data points
		x = [x[0] for x in qpVmafSorted]
		y = [x[1] for x in qpVmafSorted]
		numDataPoints = len(x)
		
		# If only one data point assume e=100 and calculate by hand
		if numDataPoints == 1:
			# y = a(x-t)^4 + e
			a = (y[0]-100.0)/((x[0]-t)**4)
			y2 = [(a*(x-t)**4 + 100.0) for x in allowableQPs]
		else:
			# Polynomial degree (list of which orders to include)
			deg = np.append(np.arange(4,(5-numDataPoints),-1),0)
			if numDataPoints > 5:
				deg = [4,3,2,1,0]
			
			# Use numpy polyfit function
			c, stats = P.polyfit(x,y,deg,full=True)
			y2 = [P.polyval(x,c) for x in allowableQPs]
			
			# Use scipy univariate spline fit function
			spline = UnivariateSpline(x,y, k=1)
			y2 = [spline(x) for x in allowableQPs]
			
		# Use qp-vmaf curve to predict QP value
		qpVmafPredictions = list(zip(allowableQPs, y2))
	
		# Select value closest to targetVMAF. Then limit the size of QP jump.
		nextQP = min(qpVmafPredictions, key=lambda l: abs(l[1] - targetVMAF))[0]
		nextQP = min(nextQP, self.nextQP + maxQpStep)
		self.nextQP = nextQP
		return 1
		
	
	def calculateNewQP3(self, targetVMAF, minQP, maxQP):
		# Make sure there is some data to process
		numDataPoints = len(self.qpVmaf)
		if numDataPoints == 0:
			print("Error: Should not have called calculateNewQP before first transcode")
			exit()
			
		# If the most recent VMAF is 100.0 add 1/3rd of total QP range to QP
		maxQpStep = int(np.ceil((maxQP-minQP)/3.0))
		if [x[1] for x in self.qpVmaf].count(100.0) == numDataPoints:
			if self.nextQP + maxQpStep <= maxQP:
				nextQP = self.nextQP + maxQpStep
			else:
				nextQP = maxQP
			self.nextQP = nextQP
			return 1
		
		# Sort qpVmaf by QP from low to high
		qpVmafSorted = sorted(self.qpVmaf, key=lambda row: row[0])
			
			
		# Modelling the qp-vmaf curve as y=100 up to an unknown (possibly negative)
		# qp number refered in the code as "t" for transition. After that the curve
		# is modelled as a 4th order polynomial:
		# a(x-t)^4 + b(x-t)^3 + c(x-t)^2 + d(x-t) + e
		
		# Initially
		t = 0
		e = 100
		
		# If there are multiple VMAF=100 entries remove all but the one with
		# the largest QP
		while [x[1] for x in qpVmafSorted].count(100.0) > 1:
			qpVmafSorted.pop(0)
			
		# If vmaf==100 is in the data, store, set "t" as that qp value then remove it
		if [x[1] for x in qpVmafSorted].count(100.0) == 1:
			t = qpVmafSorted[0][0]
			qpVmafSorted.pop(0)
		
		# At this point qpVmafSorted should only contain data from the 4th order
		# polynomial part of the qp-vmaf curve, plus we should have an initial guess
		# for "t"
			
		# Separate into x's and y's and recalculate number of data points
		x = [x[0] for x in qpVmafSorted]
		y = [x[1] for x in qpVmafSorted]
		numDataPoints = len(x)
		
		# If there are two or more data points we can have a better guess for "t" based
		# on the reduced 4th order polynomial: y = a(x-t)^4 + e
		# with two sets of data we can rearrange as:
		# [(x1-t)^4]/[(x2-t)^4] - (y1 - e)/(y2 - e) = 0
		# I then brute force guess allowable "t" values to find the value
		if numDataPoints >= 2:
			maxT = maxQP
			for rq in qpVmafSorted:
				if rq[1] < 100.0:
					maxT = rq[0]
					break
			minT = -50.0
			for rq in qpVmafSorted:
				if rq[1] == 100.0:
					minT = rq[1]
			allowableTs = np.arange(minT,maxT,0.1)
			Ys = ((y[0]-e)/(y[-1]-e))
			XTs = [((x[0]-t)**4)/(x[-1]-t)**4 - Ys for t in allowableTs]
			xpPredictions = list(zip(allowableTs, XTs))
			newT = min(xpPredictions, key=lambda l: abs(l[1]))[0]
			if newT > t:
				t = newT
			print("t = " + str(t))
		
		# "t" provides us the minimum QP we can usefully use for a new guess (anything
		# smaller would produce vmaf=100.0)
		if t > minQP:
			minQP = int(np.ceil(t))
			
		# The maximum QP can be lowered to the smallest QP in the qpVmaf array so long
		# as the VMAF is smaller than targetVMAF
		for rq in qpVmafSorted:
			if rq[1] < targetVMAF:
				maxQP = rq[0]
				break
		
		# Use minQP and maxQP to construct array of all allowable QPs
		print(minQP, maxQP)
		allowableQPs = np.arange(minQP, maxQP+1, 1)
			
		# Use previous data to construct qp-vmaf curve
		if numDataPoints <= 2:
			# y = a(x-t)^4 + e
			#if y[-1] > 95.0:
			#	y[-1] = 95.0
			a = (y[-1]-e)/((x[-1]-t)**4)
			y2 = [(a*(x-t)**4 + e) for x in allowableQPs]
		elif numDataPoints == 3:
			# y = a(x-t)^4 + b(x-t)^3 + e
			denom = ((x[2]-t)**3 * (x[1]-t)**4) - ((x[1]-t)**3 * (x[2]-t)**4)
			b = ((x[1]-t)**4 * (y[2] - e) - (x[2]-t)**4 * (y[1] - e))/denom
			a = (y[1] - e - b*(x[1]-t)**3)/(x[1]-t)**4
			y2 = [(a*(x-t)**4 + b*(x-t)**3 + e) for x in allowableQPs]
		else:
			# Scipy linear interpolation
			f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
			y2 = f(allowableQPs)
		
		# Use qp-vmaf curve to predict QP value
		qpVmafPredictions = list(zip(allowableQPs, y2))
		
		# Select value closest to targetVMAF. Then limit the size of QP jump.
		nextQP = min(qpVmafPredictions, key=lambda l: abs(l[1] - targetVMAF))[0]
		nextQP = min(nextQP, self.nextQP + maxQpStep)
		self.nextQP = nextQP
		return 1
	
	
	def calculateNewQP2(self, targetVMAF, minQP, maxQP):
		# Pre-process existing data
		numDataPoints = len(self.qpVmaf)
		if numDataPoints != 0:
			# Sort qpVmaf by QP from low to high
			qpVmafSorted = sorted(self.qpVmaf, key=lambda row: row[0])
		
			# If there are multiple VMAF=100 entries remove all but the one with
			# the largest QP
			while [x[1] for x in qpVmafSorted].count(100.0) > 1:
				qpVmafSorted.pop(0)
			
			# Separate into x's and y's
			x = [x[0] for x in qpVmafSorted]
			y = [x[1] for x in qpVmafSorted]
		else:
			print("Error: Should not have called calculateNewQP before first transcode")
			exit()
			
		# If most recent VMAF is 100.0 add 1/3rd of total QP range to QP
		if y[-1] >= 100.0:
			if self.nextQP + 10 <= maxQP:
				nextQP = self.nextQP + int(np.ceil((maxQP-minQP)/3.0))
			else:
				nextQP = maxQP
			self.nextQP = nextQP
			return 1
		
		
		# Modelling the qp-vmaf curve as y=100 up to an unknown (possibly negative)
		# qp number refered in the code as "t" for transition. After that the curve
		# is modelled as a 4th order polynomial:
		# a(x-t)^4 + b(x-t)^3 + c(x-t)^2 + d(x-t) + e
		
		# Maximum value when (x-t) == 0 is 100.0 therefore:
		e = 100.0
		
		# First, calculate "t". Before we have two results we assume t=0 as there is not
		# enough data to calculate it. This is the "simplified" calculation of "t" based
		# on the equation: a(x-t)^4 + e. I am assuming "t" will not change.
		t = 0
		if numDataPoints >= 2:
			if y[0] == 100.0 and numDataPoints == 2:
				t = x[0]
			else:
				maxT = x[0]
				lowIndex = 0
				if y[0] == 100.0:
					maxT = x[1]
					lowIndex = 1
				allowableTs = np.arange(0,maxT,0.1)
				XTs = [((x[lowIndex]-t)**4)/(x[-1]-t)**4 for t in allowableTs]
				Ys = ((y[lowIndex]-e)/(y[-1]-e))
				xpPredictions = list(zip(allowableTs, XTs))
				t = min(xpPredictions, key=lambda l: abs(l[1] - Ys))[0]
		
		# "t" provides us the minimum QP we can usefully use for a new guess (anything
		# smaller would produce vmaf=100.0)
		if t > minQP:
			minQP = int(np.ceil(t))
		allowableQPs = np.arange(minQP, maxQP+1, 1)
			
		# Use previous data to construct qp-vmaf curve
		if numDataPoints <= 2:
			# y = a(x-t)^4 + e
			if y[-1] > 95.0:
				y[-1] = 95.0
			a = (y[-1]-e)/((x[-1]-t)**4)
			y2 = [(a*(x-t)**4 + e) for x in allowableQPs]
		elif numDataPoints == 3:
			# y = a(x-t)^4 + b(x-t)^3 + e
			denom = ((x[2]-t)**3 * (x[1]-t)**4) - ((x[1]-t)**3 * (x[2]-t)**4)
			b = ((x[1]-t)**4 * (y[2] - e) - (x[2]-t)**4 * (y[1] - e))/denom
			a = (y[1] - e - b*(x[1]-t)**3)/(x[1]-t)**4
			y2 = [(a*(x-t)**4 + b*(x-t)**3 + e) for x in allowableQPs]
		else:
			# Scipy linear interpolation
			f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
			y2 = f(allowableQPs)
		
		# Use qp-vmaf curve to predict QP value
		qpVmafPredictions = list(zip(allowableQPs, y2))
		
		nextQP = min(qpVmafPredictions, key=lambda l: abs(l[1] - targetVMAF))[0]
		self.nextQP = nextQP
		return 1
			
				
	def calculateNewQP(self, targetVMAF, minQP, maxQP):
		# Array of allowable QPs
		allowableQPs = np.arange(minQP, maxQP+1, 1)
		
		# Arrange previous data
		numDataPoints = len(self.qpVmaf)
		if numDataPoints != 0:
			qpVmafSorted = sorted(self.qpVmaf, key=lambda row: row[0])
			x = [x[0] for x in qpVmafSorted]
			y = [x[1] for x in qpVmafSorted]
		
			# Prepend qp=0, vmaf=100 to previous data if there is not already a vmaf score
			# equal to or greater than 100.
			if max(y) < 100.0:
				x.insert(0,0)
				y.insert(0,100.0)
				
			# Recalculate number of datapoints
			numDataPoints = len(x)
		else:
			print("Error: Should not have called calculateNewQP before first transcode")
			exit()
		
		# Use previous data to construct qp-vmaf curve
		if numDataPoints == 1:
			if self.nextQP + 10 <= maxQP:
				nextQP = self.nextQP + 10
			else:
				nextQP = maxQP
			self.nextQP = nextQP
			return 1
		elif numDataPoints == 2:
			# y = ax^4 + c
			if y[1] > 95.0:
				y[1] = 95.0
			c = y[0]
			a = (y[1]-y[0])/(x[1]**4)
			y2 = [(a*x**4 + c) for x in allowableQPs]
		elif numDataPoints == 3:
			# y = ax^4 + bx^3 + c
			c = y[0]
			denom = (x[2]**3 * x[1]**4) - (x[1]**3 * x[2]**4)
			b = (x[1]**4 * (y[2] - c) - x[2]**4 * (y[1] - c))/denom
			a = (y[1] - c - b*x[1]**3)/x[1]**4
			y2 = [(a*x**4 + b*x**3 + c) for x in allowableQPs]
		else:
			# Scipy linear interpolation
			f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
			y2 = f(allowableQPs)
		
		# Use qp-vmaf curve to predict QP value
		qpVmafPredictions = list(zip(allowableQPs, y2))
		nextQP = min(qpVmafPredictions, key=lambda l: abs(l[1] - targetVMAF))[0]
		self.nextQP = nextQP
		return 1
		
		
	def addQpVmaf(self, QP, VMAF):
		# If neither QP nor VMAF exist in qpVmaf list then add them
		# If it is an existing QP but a new VMAF, update the VMAF value in the qpVmaf list
		newQP = True
		newVMAF = True
		oldEntry = ()
		for rq in self.qpVmaf:
			if rq[0] == QP:
				newQP = False
			if rq[1] == VMAF:
				newVMAF = False
				oldEntry = rq
		if newQP and newVMAF:
			self.qpVmaf.append((QP,VMAF))
		elif newVMAF:
			self.qpVmaf.remove(oldEntry)
			self.qpVmaf.append((QP,VMAF))
	
	
	def isOptimised(self):
		optimised = True
		if len(self.qpVmaf) == 0:
				optimised = False
		else:
			QPs = [rq[0] for rq in self.qpVmaf]
			if self.nextQP not in QPs:
				optimised = False
		return optimised


def main():
	# Store start time
	startTime = datetime.datetime.now()
	
	# Manage arguments
	parser = ArgumentParser()
	parser.add_argument("file", nargs="?")
	parser.add_argument("-q", "--quality", type=float, default=85.0)
	parser.add_argument("-e", "--encoder", type=str, default="x264")
	parser.add_argument("-g", "--guess", type=int, default=30)
	parser.add_argument("-t", "--threshold", type=float, default=30.0)
	parser.add_argument("-m", "--model", type=str, default="/usr/local/share/model/vmaf_v0.6.1.pkl")
	parser.add_argument("-s", "--subsample", type=int, default=1)
	args = parser.parse_args()
	fileName = abspath(args.file)
	
	# Check for valid quality targetVMAF
	if not 0.0 < args.quality < 100.0:
		print("Invalid target quality value: " + str(args.quality))
		print("Target quality value has to be between 0.0 and 100.0")
		print("Recommended target quality values are between 50.0 and 90.0")
		exit()
	elif not 50.0 <= args.quality <= 90.0:
		print("Target quality value " + str(args.quality) + " is valid")
		print("However a target quality value are between 50.0 and 90.0 is recommended")
	
	# Check for valid encoder
	allowableEncoders = ["x264", "x265"]
	if args.encoder not in allowableEncoders:
		print("Unknown encoder: " + str(args.encoder))
		exit()
		
	# Select correct encoder tuple (encoderName, minQP, maxQP)
	encoder = ("x264", 10, 51)
	if args.encoder == "x265":
		encoder = ("x265", 0, 61)
	
	# Check for valid initial guess
	allowableQPs = np.arange(encoder[1], encoder[2]+1, 1)
	if args.guess not in allowableQPs:
		print("Initial guess QP value " + str(args.guess) + " is invalid")
		print("Allowable values are whole numbers between " + str(encoder[1]) + " and " + str(encoder[2]))
		exit()
		
		
	# Insatiate Video class
	video = Video(fileName, encoder)
	
	# Generate and order shots
	video.generateShotList(args.threshold, args.guess)
	video.sortShots()
	
	# Optimise for quality (re-transcode until suitable VMAF achieved)
	video.optimise(args.quality, args.model, args.subsample)
	
	# Print details
	video.printSummary(args.quality)
	
	# Final transcode and quality check
	video.finalTranscode()
	video.calculateVideoVMAF(args.model)
	
	# Store end time & print total time
	endTime = datetime.datetime.now()
	print("Encoding time: " + str(endTime-startTime))
	
if __name__ == "__main__":
	main()

