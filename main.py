#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time
import os
import csv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_motor.msg import xycar_motor
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Red color HSV range
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 100, 100])
UPPER_RED2 = np.array([180, 255, 255])

# Desired target distance
TARGET_DISTANCE = 30.0

# Settings for calculating average error
ERROR_RECORD_DURATION = 3.0  # seconds
LOOP_RATE = 10  # Hz
SAMPLE_SIZE = int(ERROR_RECORD_DURATION * LOOP_RATE)  # 3 seconds * 10 Hz = 30 samples

# Image and target conditions
IMG_HEIGHT = 240
DESIRED_CENTER = 160
VALID_Y_MIN = 40
VALID_Y_MAX = 200

# Control-related defaults
BASE_SPEED = 19

# PID controller parameters
KP_STEER = 0.05
KD_STEER = 0.1
KI_STEER = 0.01
KP_SPEED = 0.8
KD_SPEED = 0.1
KI_SPEED = 0.01

# Global variables
bridge = CvBridge()
cv_image = np.empty(shape=[0])
motor_pub = None

# Lists for recording errors
abs_steering_errors = []
abs_speed_errors = []
steering_errors = []
speed_errors = []

# Control variables
integral_speed_error = 0.0
integral_steer_error = 0.0
previous_steer_error = 0.0
previous_speed_error = 0.0
previous_time = time.time()
previous_angle = 0.0
previous_speed_cmd = 0.0

# Variables for plotting
steering_error_history = []
speed_error_history = []
abs_steering_error_history = []
abs_speed_error_history = []
time_history = []

def print_status(distance, steer_error, speed_error, avg_abs_steer, avg_abs_speed, angle, speed):
    """Print structured status information to terminal"""
    print("\n" + "="*50)
    print("Current Status:")
    print("-"*50)
    print("Distance Between Dots: {:.2f} pixels".format(distance))
    print("Errors:")
    print("  - Steering Error: {:.2f}".format(steer_error))
    print("  - Speed Error: {:.2f}".format(speed_error))
    print("Average Absolute Errors (3s window):")
    print("  - Steering: {:.2f}".format(avg_abs_steer))
    print("  - Speed: {:.2f}".format(avg_abs_speed))
    print("Control Commands:")
    print("  - Angle: {:.2f}".format(angle))
    print("  - Speed: {:.2f}".format(speed))
    print("="*50)

def cam_exposure(value):
    command = 'v4l2-ctl -d /dev/videoCAM -c exposure_absolute=' + str(value)
    os.system(command)

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def detect_red_dots(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.medianBlur(red_mask, 5)
    
    y_coords, x_coords = np.where(red_mask > 0)
    
    if len(x_coords) > 0:
        left_x = np.min(x_coords)
        right_x = np.max(x_coords)
        center_x = (left_x + right_x) / 2.0
        center_y = np.mean(y_coords)
        distance = right_x - left_x
        return True, left_x, right_x, center_x, center_y, distance
    else:
        return False, 0, 0, 0, 0, 0

def compute_steering_error(center_x):
    return center_x - DESIRED_CENTER

def compute_speed_error(distance):
    return TARGET_DISTANCE - distance

def compute_average_error(errors_list):
    if len(errors_list) == 0:
        return 0.0
    return sum(errors_list) / float(len(errors_list))

def pid_control_steering(error, prev_error, integral_error, dt):
    p_term = KP_STEER * error
    d_term = KD_STEER * (error - prev_error) / dt if dt > 0 else 0
    integral_error += error * dt
    i_term = KI_STEER * integral_error
    pid_output = p_term + d_term + i_term
    
    pid_output = max(min(pid_output, 1.0), -1.0)

    if pid_output == 0:
        angle = 0
    elif pid_output > 0:
        angle = 8 + 92.0 * pid_output
        angle = min(angle, 100)
    else:
        angle = -8 - 92.0 * abs(pid_output)
        angle = max(angle, -100)

    return angle, integral_error

def pid_control_speed(error, prev_error, integral_error, dt):
    p_term = KP_SPEED * error
    d_term = KD_SPEED * (error - prev_error) / dt if dt > 0 else 0
    integral_error += error * dt
    i_term = KI_SPEED * integral_error
    speed_input = p_term + d_term + i_term
    
    speed = max(min(speed_input, 100), BASE_SPEED)
    return speed, integral_error

def drive(angle, speed):
    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor_pub.publish(motor_msg)

if __name__ == '__main__':
    rospy.init_node('final_project_node', anonymous=True)
    motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")

    cam_exposure(50)
    rate = rospy.Rate(LOOP_RATE)

    try:
        while not rospy.is_shutdown():
            if cv_image.size != 0:
                ret, left_x, right_x, center_x, center_y, distance = detect_red_dots(cv_image)

                current_time = time.time()
                dt = current_time - previous_time
                previous_time = current_time

                angle = previous_angle
                speed_cmd = previous_speed_cmd

                if ret and (VALID_Y_MIN <= center_y <= VALID_Y_MAX):
                    # Calculate errors
                    steer_error = compute_steering_error(center_x)
                    spd_error = compute_speed_error(distance)
                    
                    # Store absolute errors
                    abs_steering_errors.append(abs(steer_error))
                    abs_speed_errors.append(abs(spd_error))
                    
                    if len(abs_steering_errors) > SAMPLE_SIZE:
                        abs_steering_errors.pop(0)
                    if len(abs_speed_errors) > SAMPLE_SIZE:
                        abs_speed_errors.pop(0)

                    # Calculate averages
                    avg_abs_steer = compute_average_error(abs_steering_errors)
                    avg_abs_speed = compute_average_error(abs_speed_errors)

                    # PID control
                    normalized_steer_error = steer_error / 160.0
                    angle, integral_steer_error = pid_control_steering(normalized_steer_error, 
                                                                     previous_steer_error/160.0, 
                                                                     integral_steer_error, dt)
                    speed_cmd, integral_speed_error = pid_control_speed(spd_error, 
                                                                      previous_speed_error, 
                                                                      integral_speed_error, dt)

                    previous_steer_error = steer_error
                    previous_speed_error = spd_error

                    # Print structured status
                    print_status(distance, steer_error, spd_error, 
                               avg_abs_steer, avg_abs_speed, 
                               angle, speed_cmd)

                    # Record for plotting
                    steering_error_history.append(steer_error)
                    speed_error_history.append(spd_error)
                    abs_steering_error_history.append(abs(steer_error))
                    abs_speed_error_history.append(abs(spd_error))
                    time_history.append(current_time)

                else:
                    print("\nNo valid dots detected or out of Y-range")
                    print("Keeping previous steering angle: {:.2f}".format(angle))

                drive(angle, speed_cmd)
                
                previous_angle = angle
                previous_speed_cmd = speed_cmd

            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        # Print final statistics
        final_avg_abs_steer = compute_average_error(abs_steering_errors)
        final_avg_abs_speed = compute_average_error(abs_speed_errors)
        print("\nFinal Statistics (Last 3 seconds):")
        print("-"*50)
        print("Average Absolute Steering Error: {:.2f}".format(final_avg_abs_steer))
        print("Average Absolute Speed Error: {:.2f}".format(final_avg_abs_speed))

        # save time_history, steering_error_history, speed_error_history, abs_steering_error_history, abs_speed_error_history
        csv_filename = "error_data.csv"
        with open(csv_filename, mode='wb', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time", "Steering Error", "Abs Steering Error", "Speed Error", "Abs Speed Error"])
            for i in range(len(time_history)):
                writer.writerow([
                    time_history[i],
                    steering_error_history[i] if i < len(steering_error_history) else "",
                    abs_steering_error_history[i] if i < len(abs_steering_error_history) else "",
                    speed_error_history[i] if i < len(speed_error_history) else "",
                    abs_speed_error_history[i] if i < len(abs_speed_error_history) else ""
                ])
        print(f"\ save success:'{csv_filename}' ")

        # Plot errors
        plt.figure(figsize=(12,8))
        
        plt.subplot(221)
        plt.plot(time_history, steering_error_history)
        plt.xlabel('Time')
        plt.ylabel('Steering Error')
        plt.title('Original Steering Error Over Time')

        plt.subplot(222)
        plt.plot(time_history, abs_steering_error_history)
        plt.xlabel('Time')
        plt.ylabel('Absolute Steering Error')
        plt.title('Absolute Steering Error Over Time')

        plt.subplot(223)
        plt.plot(time_history, speed_error_history)
        plt.xlabel('Time')
        plt.ylabel('Speed Error')
        plt.title('Original Speed Error Over Time')

        plt.subplot(224)
        plt.plot(time_history, abs_speed_error_history)
        plt.xlabel('Time')
        plt.ylabel('Absolute Speed Error')
        plt.title('Absolute Speed Error Over Time')
        
        plt.tight_layout()
        print("\nDisplaying graphs. Press 'a' in the terminal window to close...")
        
        plt.show(block=False)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                plt.close('all')
                cv2.destroyAllWindows()
                break
