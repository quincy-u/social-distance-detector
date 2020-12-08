#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import time
import argparse


def find_mid_points(box):
    # box = [x_left_top, y_left_top, width, height]
    x = box[0] + box[2] * 0.5
    y = box[1] + box[3]
    middle_point = np.array([[[int(x), int(y)]]], dtype="float32")
    return middle_point

# Function to calculate bottom center for all bounding boxes and transform prespective for all points.
def get_transformed_points(boxes, transformation_matrix):

    image_pixels = []
    for box in boxes:
        coor = find_mid_points(box)
        #pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+(box[3]*0.5))]]] , dtype="float32")
        new_coor = cv2.perspectiveTransform(coor, transformation_matrix)[0][0]
        image_pixel = [int(new_coor[0]), int(new_coor[1])]
        image_pixels.append(image_pixel)

    return image_pixels

# Function calculates distance between two points(humans). distance_w, distance_h represents number
# of pixels in 180cm length horizontally and vertically. We calculate horizontal and vertical
# distance in pixels for two points and get ratio in terms of 180 cm distance using distance_w, distance_h.
# Then we calculate how much cm distance is horizontally and vertically and then using pythagoras
# we calculate distance between points in terms of cm. 
# def cal_dis(p1, p2, distance_w, distance_h):

#     h = abs(p2[1]-p1[1])
#     w = abs(p2[0]-p1[0])

#     dis_w = float((w/distance_w)*180)
#     dis_h = float((h/distance_h)*180)

#     return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

# Calculates distance between all pairs, if
def get_distances(boxes1, image_pixels, distance_w, distance_h):

    distances_matrix = []
    colored_boxes = []
    high_risk = 0
    mid_risk = 1
    low_risk = 2

    for i in range(len(image_pixels)):
        for j in range(len(image_pixels)):
            if i != j:
#                 dist = cal_dis(image_pixels[i], image_pixels[j], distance_w, distance_h)
                distance = int(np.sqrt(((float((abs(image_pixels[i][1]-image_pixels[j][1])/distance_h)*180))**2) + ((float((abs(image_pixels[i][0]-image_pixels[j][0])/distance_w)*180))**2)))

                if distance <= 100:

                    distances_matrix.append([image_pixels[i], image_pixels[j], high_risk])
                    colored_boxes.append([boxes1[i], boxes1[j], high_risk])
                elif distance > 100 and distance <=180:

                    distances_matrix.append([image_pixels[i], image_pixels[j], mid_risk])
                    colored_boxes.append([boxes1[i], boxes1[j], mid_risk])
                else:

                    distances_matrix.append([image_pixels[i], image_pixels[j], low_risk])
                    colored_boxes.append([boxes1[i], boxes1[j], low_risk])

    return distances_matrix, colored_boxes



def count_risk(distances_matrix):
    """
    Count for humans at high risk, low risk and no risk
    :param distances_matrix: a list contain [coordinate1, coordinate2, risk_level] for each close pair
    :return: a Tuple of 3 (count_of_high_risk, count_of_low_risk, count_of_no_risk)
    """

    high_risk = []
    low_risk = []
    no_risk = []

    # add the corresponding people to certain level risk list
    # if one of the pair is already counted, skip it
    for i in range(len(distances_matrix)):
        if distances_matrix[i][2] == 0:
            if not (distances_matrix[i][0] in high_risk or distances_matrix[i][0] in no_risk or distances_matrix[i][0] in low_risk):
                high_risk.append(distances_matrix[i][0])
            if not (distances_matrix[i][1] in high_risk or distances_matrix[i][1] in no_risk or distances_matrix[i][1] in low_risk):
                high_risk.append(distances_matrix[i][1])

    for i in range(len(distances_matrix)):
        if distances_matrix[i][2] == 1:
            if not (distances_matrix[i][0] in high_risk or distances_matrix[i][0] in no_risk or distances_matrix[i][0] in low_risk):
                low_risk.append(distances_matrix[i][0])
            if not (distances_matrix[i][1] in high_risk or distances_matrix[i][1] in no_risk or distances_matrix[i][1] in low_risk):
                low_risk.append(distances_matrix[i][1])

    for i in range(len(distances_matrix)):
        if distances_matrix[i][2] == 2:
            if not (distances_matrix[i][0] in high_risk or distances_matrix[i][0] in no_risk or distances_matrix[i][0] in low_risk):
                no_risk.append(distances_matrix[i][0])
            if not (distances_matrix[i][1] in high_risk or distances_matrix[i][1] in no_risk or distances_matrix[i][1] in low_risk):
                no_risk.append(distances_matrix[i][1])
    return len(high_risk), len(low_risk), len(no_risk)


# transform the selected region to bird's eye view
def transform_frame(frame, transformation_matrix):
    rows, cols, _ = frame.shape
    new_frame = cv2.warpPerspective(frame, transformation_matrix, (cols, rows))
    scale_w = int(new_frame.shape[0] / frame.shape[0])
    scale_h = int(new_frame.shape[1] / frame.shape[1])
    return new_frame, scale_w, scale_h


# Draw the Bird Eye View for region selected. Red, Orange, Green points represents different risk levels to human.
# Red: High Risk, Orange: Low Risk, Green: No Risk
def bird_eye_view(frame, distances_matrix, bottom_points, risk_count, transformation_matrix):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    orange = (0, 165, 255)
    white = (200, 200, 200)

    new_frame, scale_w, scale_h = transform_frame(frame, transformation_matrix)

#     scale_w =int(new_frame.shape[0] / frame.shape[0])
#     scale_h = int(new_frame.shape[1] / frame.shape[1])



#     blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
#     blank_image[:] = white
#     warped_pts = []
    r = []
    g = []
    o = []
    for i in range(len(distances_matrix)):

        if distances_matrix[i][2] == 0:
            if (distances_matrix[i][0] not in r) and (distances_matrix[i][0] not in g) and (distances_matrix[i][0] not in o):
                r.append(distances_matrix[i][0])
            if (distances_matrix[i][1] not in r) and (distances_matrix[i][1] not in g) and (distances_matrix[i][1] not in o):
                r.append(distances_matrix[i][1])

            new_frame = cv2.line(new_frame, (int(distances_matrix[i][0][0] * scale_w), int(distances_matrix[i][0][1] * scale_h)), (int(distances_matrix[i][1][0] * scale_w), int(distances_matrix[i][1][1]* scale_h)), red, 2)

    for i in range(len(distances_matrix)):

        if distances_matrix[i][2] == 1:
            if (distances_matrix[i][0] not in r) and (distances_matrix[i][0] not in g) and (distances_matrix[i][0] not in o):
                o.append(distances_matrix[i][0])
            if (distances_matrix[i][1] not in r) and (distances_matrix[i][1] not in g) and (distances_matrix[i][1] not in o):
                o.append(distances_matrix[i][1])

            new_frame = cv2.line(new_frame, (int(distances_matrix[i][0][0] * scale_w), int(distances_matrix[i][0][1] * scale_h)), (int(distances_matrix[i][1][0] * scale_w), int(distances_matrix[i][1][1]* scale_h)), orange, 2)

    for i in range(len(distances_matrix)):

        if distances_matrix[i][2] == 2:
            if (distances_matrix[i][0] not in r) and (distances_matrix[i][0] not in g) and (distances_matrix[i][0] not in o):
                g.append(distances_matrix[i][0])
            if (distances_matrix[i][1] not in r) and (distances_matrix[i][1] not in g) and (distances_matrix[i][1] not in o):
                g.append(distances_matrix[i][1])

    for i in bottom_points:
        new_frame = cv2.circle(new_frame, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 5)
    for i in o:
        new_frame = cv2.circle(new_frame, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, orange, 5)
    for i in r:
        new_frame = cv2.circle(new_frame, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 5)

    # pad = np.full((100,blank_image.shape[1],3), [110, 110, 100], dtype=np.uint8)
    # cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)
    # cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # blank_image = np.vstack((blank_image,pad))

    return new_frame


# Draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Orange: Low Risk
# Green: No Risk 
def social_distancing_view(frame, distances_matrix, boxes, risk_count):

    red = (0, 0, 255)
    green = (0, 255, 0)
    orange = (0,165,255)

    for i in range(len(boxes)):

        x,y,w,h = boxes[i][:]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)

    for i in range(len(distances_matrix)):

        per1 = distances_matrix[i][0]
        per2 = distances_matrix[i][1]
        closeness = distances_matrix[i][2]

        if closeness == 1:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),orange,2)

            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),orange,2)

            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),orange, 2)

    for i in range(len(distances_matrix)):

        per1 = distances_matrix[i][0]
        per2 = distances_matrix[i][1]
        closeness = distances_matrix[i][2]

        if closeness == 0:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)

            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)

            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)

    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 1)
    cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    frame = np.vstack((frame,pad))

    return frame




# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

# Function will be called on mouse events                                                          

def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
#         print("Point detected")
#         print(mouse_pts)



def calculate_social_distancing(vid_path, net, output_dir, output_vid, ln1):

    points = []
    global image
    count = 0
    vs = cv2.VideoCapture(vid_path)

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    (success, frame) = vs.read()
    (H, W) = frame.shape[:2]

    if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break

            points = mouse_pts


    # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
    # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
    # This bird eye view then has the property property that points are distributed uniformally horizontally and 
    # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
    # equally distributed, which was not case for normal view.

    src = np.float32(np.array(points[:4]))
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    prespective_transform = cv2.getPerspectiveTransform(src, dst)
    new_frame, scale_w, scale_h = transform_frame(frame, prespective_transform)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height+140))
#     video_width, video_height = img.shape[1], img.shape[0]
#     output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (video_width, video_height))
    bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))


    while True:

        (success, frame) = vs.read()

        if not success:
#             print('here')
            break

        (H, W) = frame.shape[:2]


        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

    ####################################################################################

        # YOLO v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]

        if len(boxes1) == 0:
            count = count + 1
            continue

        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = get_transformed_points(boxes1, prespective_transform)

        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = count_risk(distances_mat)

        frame1 = np.copy(frame)

        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
        bird_image = bird_eye_view(frame, distances_mat, person_points, risk_count, prespective_transform)
        img = social_distancing_view(frame1, bxs_mat, boxes1, risk_count)

        # Show/write image and videos
        if count != 0:
            output_movie.write(img)
            bird_movie.write(bird_image)

            cv2.imshow('Bird Eye View', img)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()





# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

# Function will be called on mouse events                                                          

def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4: # points for ROI
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else: # points to define 6 deet
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
#         print("Point detected")
#         print(mouse_pts)



def calculate_social_distancing4colab(vid_path, net, output_dir, output_vid, ln1):

    points = []
    global image
    count = 0
    vs = cv2.VideoCapture(vid_path)

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    (success, frame) = vs.read()
    (H, W) = frame.shape[:2]


    points = mouse_pts


    # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
    # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
    # This bird eye view then has the property property that points are distributed uniformally horizontally and 
    # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
    # equally distributed, which was not case for normal view.

    src = np.float32(np.array(points[:4]))
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    prespective_transform = cv2.getPerspectiveTransform(src, dst)
    new_frame, scale_w, scale_h = transform_frame(frame, prespective_transform)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height+140))
#     video_width, video_height = img.shape[1], img.shape[0]
#     output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (video_width, video_height))
    bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))


    while True:

        (success, frame) = vs.read()

        if not success:
#             print('here')
            break

        (H, W) = frame.shape[:2]


        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

    ####################################################################################

        # YOLO v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]

        if len(boxes1) == 0:
            count = count + 1
            continue

        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = get_transformed_points(boxes1, prespective_transform)

        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = count_risk(distances_mat)

        frame1 = np.copy(frame)

        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
        bird_image = bird_eye_view(frame, distances_mat, person_points, risk_count, prespective_transform)
        img = social_distancing_view(frame1, bxs_mat, boxes1, risk_count)

        # Show/write image and videos
        if count != 0:
            output_movie.write(img)
            bird_movie.write(bird_image)

            cv2_imshow(img)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()


# In[23]:


def main(output_dir="./output/", output_vid="./output_vid/", video_path="data/example2.mp4",
         weights_path="models/yolov3.weights", config_path="models/yolov3.cfg"):
    """
    :param output_dir: the path of output video
    :param output_vid: the path of output bird's eye view video
    :param video_path: the path of input video
    :param weights_path: Yolov3 weights path
    :param config_path: Yolov3 config path
    :return:
    """
    global mouse_pts
    mouse_pts = []
    net_yl = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # set mouse callback 

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(62)

    calculate_social_distancing(video_path, net_yl, output_dir, output_vid, ln1)


if __name__ == "__main__":
    confid = 0.5
    thresh = 0.5
    mouse_pts = []
    main(video_path = "data/example.mp4")





