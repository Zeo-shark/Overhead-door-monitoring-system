from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import datetime

t0 = time.time()

def run():


	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())


	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	#Uolo v3 MobilenetSSD
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)


	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])


	writer = None


	W = None
	H = None



	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}


	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]


	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)


	while True:

		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame


		if args["input"] is not None and frame is None:
			break



		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)


		status = "Waiting"
		rects = []


		if totalFrames % args["skip_frames"] == 0:

			status = "Detecting"
			trackers = []


			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):

				confidence = detections[0, 0, i, 2]


				if confidence > args["confidence"]:

					idx = int(detections[0, 0, i, 1])


					if CLASSES[idx] != "person":
						continue


					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# construct a dlib rectangle object from the bounding

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)


					trackers.append(tracker)


		else:

			for tracker in trackers:

				status = "Tracking"


				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				rects.append((startX, startY, endX, endY))


		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
					(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)



		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():

			to = trackableObjects.get(objectID, None)


			if to is None:
				to = TrackableObject(objectID, centroid)


			else:

				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)


				if not to.counted:

					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True


					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						#print(empty1[-1])
						x = []

						x.append(len(empty1)-len(empty))
						#print("Total people inside:", x)

						if sum(x) >= config.Threshold:
							cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config.ALERT:
								print("[INFO] Sending email alert..")
								Mailer().send(config.MAIL)
								print("[INFO] Alert sent")

						to.counted = True



			trackableObjects[objectID] = to


			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]


		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)



		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF


		if key == ord("q"):
			break


		totalFrames += 1
		fps.update()

		if config.Timer:

			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break


	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



	# if not args.get("input", False):
	# 	vs.stop()
	#
	# # otherwise, release the video file pointer
	# else:
	# 	vs.release()


	cv2.destroyAllWindows()



if config.Scheduler:

	schedule.every().day.at("9:00").do(run)

	while 1:
		schedule.run_pending()

else:
	run()
