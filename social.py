#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import time
import argparse


class box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.risk = 2  # default risk level for each person is green

    def set_risk(self, risk):
        self.risk = risk

    def parameters(self):
        return self.x, self.y, self.w, self.h



def find_mid_points(box):
    '''
    find the bottom mid point of the box
    :param box: the box object
    :return: return the coordinate the bottom mid point
    '''
    x = box.x + box.w * 0.5
    y = box.y + box.h
    middle_point = np.array([[[int(x), int(y)]]], dtype="float32")
    return middle_point


def get_transformed_points(boxes, transformation_matrix):
    '''
    get the transformed points

    :param boxes: the list of boxes for each detected people
    :param transformation_matrix: the transformation matrix
    :return: the transformed point of each person
    '''

    image_pixels = []
    for box in boxes:
        coor = find_mid_points(box)
        new_coor = cv2.perspectiveTransform(coor, transformation_matrix)[0][0]
        image_pixel = [int(new_coor[0]), int(new_coor[1])]
        image_pixels.append(image_pixel)

    return image_pixels


def get_distances(boxes, person_points, distance_w, distance_h):
    """
    Calculates distance between all pairs, if they are close, give them risk level according to the distance

    :param boxes: boxes of each recognized Pedestrian in the image
    :param person_points: Pedestrians' coordinates after transformation
    :param distance_w: number of pixels in 6 ft length horizontally
    :param distance_h: number of pixels in 6 ft length vertically
    :return: a tuple of 3.  First is all the pairs of boxes after transformation, the second is all the pairs before the
             transformation. The last one is the set of all the boxes
    """
    distance_lst = []
    colored_pairs = []
    colored_boxes = []
    high_risk = 0
    low_risk = 1
    safe = 2

    for i in range(len(person_points)):
        for j in range(len(person_points)):
            if i != j:
                # calculate the euclidean distance and normalize it to 6 ft which is 180cm.
                distance = int( 180 * np.sqrt(
                    ((float((abs(person_points[i][1] - person_points[j][1]) / distance_h))) ** 2) + (
                            (float((abs(person_points[i][0] - person_points[j][0]) / distance_w))) ** 2)))

                if distance <= 180:
                    risk_lvl = high_risk
                elif 180 < distance <= 230:
                    risk_lvl = low_risk
                else:
                    risk_lvl = safe

                distance_lst.append([person_points[i], person_points[j], risk_lvl])
                colored_pairs.append([boxes[i], boxes[j], risk_lvl])

                # generate the set of all the boxes.
                # Only update the risk level of a person if the current risk lvl is higher than the stored one.
                for box in [boxes[i], boxes[j]]:
                    if box not in colored_boxes:
                        colored_boxes.append(box)
                    box.set_risk(min(risk_lvl, box.risk))
    return distance_lst, colored_pairs, colored_boxes


def transform_frame(frame, transformation_matrix):
    """
    transform the selected region to bird's eye view

    :param frame: the original image
    :param transformation_matrix: the transformation matrix
    :return: the image after transform, width scale, and height scale
    """
    rows, cols, _ = frame.shape
    new_frame = cv2.warpPerspective(frame, transformation_matrix, (cols, rows))
    scale_w = int(new_frame.shape[0] / frame.shape[0])
    scale_h = int(new_frame.shape[1] / frame.shape[1])
    return new_frame, scale_w, scale_h

def homography(src, dst):
    '''
    use src and dst point to calculate a homography transformation matrix

    :param src: the source points
    :param dst: the destination points
    :return: a transformation matrix
    '''
    Ax = []
    for i in range(0, len(src)):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        Ax.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        Ax.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    Ax = np.asarray(Ax)
    U, S, Vh = np.linalg.svd(Ax)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

# Draw the Bird Eye View for region selected. Red, Yellow, Green points represents different risk levels to human.
# Red: High Risk, Yellow: Low Risk, Green: No Risk
def bird_eye_view(frame, distances_matrix, bottom_points, transformation_matrix):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)

    new_frame, scale_w, scale_h = transform_frame(frame, transformation_matrix)

    r = []
    g = []
    o = []
    for i in range(len(distances_matrix)):

        if distances_matrix[i][2] == 0:
            if (distances_matrix[i][0] not in r) and (distances_matrix[i][0] not in g) and (distances_matrix[i][0] not in o):
                r.append(distances_matrix[i][0])
            if (distances_matrix[i][1] not in r) and (distances_matrix[i][1] not in g) and (distances_matrix[i][1] not in o):
                r.append(distances_matrix[i][1])

            new_frame = cv2.line(new_frame, (int(distances_matrix[i][0][0] * scale_w), int(distances_matrix[i][0][1] * scale_h)), (int(distances_matrix[i][1][0] * scale_w), int(distances_matrix[i][1][1] * scale_h)), red, 2)

    for i in range(len(distances_matrix)):

        if distances_matrix[i][2] == 1:
            if (distances_matrix[i][0] not in r) and (distances_matrix[i][0] not in g) and (distances_matrix[i][0] not in o):
                o.append(distances_matrix[i][0])
            if (distances_matrix[i][1] not in r) and (distances_matrix[i][1] not in g) and (distances_matrix[i][1] not in o):
                o.append(distances_matrix[i][1])

            new_frame = cv2.line(new_frame, (int(distances_matrix[i][0][0] * scale_w), int(distances_matrix[i][0][1]* scale_h)), (int(distances_matrix[i][1][0] * scale_w), int(distances_matrix[i][1][1] * scale_h)), yellow, 2)

    for i in range(len(distances_matrix)):

        if distances_matrix[i][2] == 2:
            if (distances_matrix[i][0] not in r) and (distances_matrix[i][0] not in g) and (distances_matrix[i][0] not in o):
                g.append(distances_matrix[i][0])
            if (distances_matrix[i][1] not in r) and (distances_matrix[i][1] not in g) and (distances_matrix[i][1] not in o):
                g.append(distances_matrix[i][1])

    for i in bottom_points:
        new_frame = cv2.circle(new_frame, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 5)
    for i in o:
        new_frame = cv2.circle(new_frame, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 5)
    for i in r:
        new_frame = cv2.circle(new_frame, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 5)

    return new_frame


# Draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk
def social_distancing_view(frame, colored_pairs, colored_boxes):
    '''
    Draw the boxes and lines on the current frame

    :param frame: the initial clean frame
    :param colored_pairs: all the paris of the boxes in the form of (box1, box2, risk_level)
    :param colored_boxes: the set of all the boxes with parameters set correctly
    :return: the final frame with drawn boxes and lines on it
    '''

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0,165,255)

    high_risk_count = 0
    low_risk_count = 0
    safe_count = 0

    # Draw the box for each detected person
    for box in colored_boxes:
        risk_lvl = box.risk

        if risk_lvl == 0:
            sign = 'Danger'
            color = red
            high_risk_count += 1
        elif risk_lvl == 1:
            sign = 'Careful'
            color = yellow
            low_risk_count += 1
        else:
            sign = 'Safe'
            color = green
            safe_count += 1
        x, y, w, h = box.parameters()
        cv2.putText(frame, sign, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


    # Draw lines between boxes if the social distance is risky
    for i in range(len(colored_pairs)):
        box1 = colored_pairs[i][0]
        box2 = colored_pairs[i][1]
        risk_lvl = colored_pairs[i][2]

        if risk_lvl != 2:
            if risk_lvl == 1:
                color = yellow
            else:
                color = red
            x1,y1,w1,h1 = box1.parameters()
            x2,y2,w2,h2 = box2.parameters()
            frame = cv2.line(frame, (int(x1+w1/2), int(y1+h1/2)), (int(x2+w2/2), int(y2+h2/2)), color, 2)

    pad = np.full((140,frame.shape[1],3), [255,204,204], dtype=np.uint8)
    font_size = 0.8
    font_thickness = 2
    cv2.putText(pad, "Count of people:", (20, 30), font, font_size, (0, 51, 102), font_thickness)
    cv2.putText(pad, "overly close :(", (100, 70), font, font_size, red, font_thickness)
    cv2.putText(pad, "a little close :|", (350, 70), font, font_size, yellow, font_thickness)
    cv2.putText(pad, "safe :)", (600, 70), font, font_size, green, font_thickness)
    cv2.putText(pad, str(high_risk_count), (180, 110), font, font_size, red, font_thickness)
    cv2.putText(pad, str(low_risk_count), (430, 110), font, font_size, yellow, font_thickness)
    cv2.putText(pad, str(safe_count) , (680, 110), font, font_size, green, font_thickness)
    # cv2.putText(pad, "Count of people: " + str(safe_count) + " ;", (50, 100), font, 0.6, (255, 255, 255), 1)
    # cv2.putText(pad, "# of people staying a little close: " + str(safe_count) + " ;", (50, 80), font, 0.6, (255,255,255), 1)
    # cv2.putText(pad, "# of people staying over close: " + str(high_risk_count) + " ;", (50, 60), font, 0.6, (255,255,255), 1)
    frame = np.vstack((frame,pad))

    return frame


def calculate_social_distancing(vid_path, net, output_dir, ln1):

    points = []
    global image
    global font
    font = cv2.FONT_HERSHEY_SIMPLEX
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
    # prespective_transform, _ = cv2.findHomography(src, dst)
    # prespective_transform = cv2.getPerspectiveTransform(src, dst)
    prespective_transform = homography(src, dst)
    new_frame, scale_w, scale_h = transform_frame(frame, prespective_transform)

    file_name = vid_path.split('/')[-1].split('.')[0]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter("./output_vid/{}_distancing.avi".format(file_name), fourcc, fps, (width, height+140))
    bird_movie = cv2.VideoWriter("./output_vid/{}_bird_eye_view.avi".format(file_name), fourcc, fps, (int(width * scale_w), int(height * scale_h)))

    fps_time = time.time()
    while True:

        (success, frame) = vs.read()

        if not success:
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
        layerOutputs = net.forward(ln1)
        boxes = []
        boxes_4_cv = []
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

                        rectangle = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = rectangle.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes_4_cv.append([x, y, int(width), int(height)])
                        boxes.append(box(x, y, int(width), int(height)))
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes_4_cv, confidences, confid, thresh)

        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])

        if len(boxes1) == 0:
            count = count + 1
            continue

        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = get_transformed_points(boxes1, prespective_transform)

        # Here we will calculate distance between transformed points(humans)
        distances_mat, colored_pairs, colored_boxes = get_distances(boxes1, person_points, distance_w, distance_h)
        # risk_count = count_risk(distances_mat)

        frame1 = np.copy(frame)

        # Draw bird eye view and frame with bouding boxes around humans according to risk factor
        bird_image = bird_eye_view(frame, distances_mat, person_points, prespective_transform)
        img = social_distancing_view(frame1, colored_pairs, colored_boxes)

        # Write Fps
        temp = time.time()
        img_row, img_col,_ = img.shape
        cv2.putText(img, "FPS: " + str(int(1/(temp - fps_time))), (int(img_col * 0.85), int(img_row * 0.95)), font, 0.8, (255,255,255), 2)
        fps_time = time.time()


        # Show/write image and videos
        if count != 0:
            output_movie.write(img)
            bird_movie.write(bird_image)

            cv2.imshow('Social distance', img)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()

def get_mouse_points(event, x, y, flags, param):
    """
    The callback function of cv2 mouse click
    Add all points to the global variable mouse_pts.
    First 4 points form two parallel lines in the real world.
    i.e. (Top left, Bottom left, Bottom right, Top right)
    The last 3 points form two orthogonal lines in the real world.
    Point 5 and 6 should form horizontal line and point 5 and 7 should form vertical line.
    The length of those two lines should be the safe distance which is 6 ft.
    """
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            # points for region selected of bird's eye view
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        else:
            # points to define safe distance
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        # draw a line to better visualize
        if 1 <= len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))


def main(output_dir="./output/", output_vid="./output_vid/", video_path="data/example1.mp4",
         weights_path="models/yolov4-tiny-pedestrian_last.weights", config_path="models/yolov4-tiny-pedestrian.cfg"):
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

    calculate_social_distancing(video_path, net_yl, output_dir, ln1)


if __name__ == "__main__":
    confid = 0.05
    thresh = 0.1
    mouse_pts = []
    main(video_path="data/example.mp4")
    # data_lst = ["data/example.mp4", "data/example1.mp4", "data/example2.mp4", "data/example3.mp4"]
    # for data in data_lst:
    #     main(video_path=data)
