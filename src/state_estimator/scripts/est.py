#!/usr/bin/env python

# Columbia Engineering
# MECS 4602 - Fall 2018

import math
import numpy
import time

import rospy

from state_estimator.msg import RobotPose
from state_estimator.msg import SensorData

class Estimator(object):
    def __init__(self):

        # Publisher to publish state estimate
        self.pub_est = rospy.Publisher("/robot_pose_estimate", RobotPose, queue_size=1)

        # Initial estimates for the state and the covariance matrix
        self.x = numpy.zeros((3,1))
        self.P = numpy.zeros((3,3))

        # Covariance matrix for process (model) noise
        self.V = numpy.zeros((3,3))
        self.V[0,0] = 0.0025
        self.V[1,1] = 0.0025
        self.V[2,2] = 0.005

        self.step_size = 0.01

        # Subscribe to command input and sensory output of robot
        rospy.Subscriber("/sensor_data", SensorData, self.sensor_callback)
        
    # This function gets called every time the robot publishes its control 
    # input and sensory output. You must make use of what you know about 
    # extended Kalman filters to come up with an estimate of the current
    # state of the robot and covariance matrix.
    # The SensorData message contains fields 'vel_trans' and 'vel_ang' for
    # the commanded translational and rotational velocity respectively. 
    # Furthermore, it contains a list 'readings' of the landmarks the
    # robot can currently observe
    def estimate(self, sens):

        #### ----- YOUR CODE GOES HERE ----- ####
        u = numpy.zeros((2,1))
        u[0] = sens.vel_trans
        u[1] = sens.vel_ang
        
        x_k = self.x
        
        F = numpy.zeros((3,3))
        for i in range(3):
            F[i][i] = 1
        F[0][2] = -self.step_size*u[0]*math.sin(x_k[2])
        F[1][2] = self.step_size*u[0]*math.cos(x_k[2])

        x_k1_hat = numpy.zeros((3,1))
        x_k1_hat[0] = x_k[0] + self.step_size*u[0]*math.cos(x_k[2])
        x_k1_hat[1] = x_k[1] + self.step_size*u[0]*math.sin(x_k[2])
        x_k1_hat[2] = x_k[2] + self.step_size*u[1]

        P_k1_hat = numpy.dot(numpy.dot(F,self.P),numpy.transpose(F)) + self.V
        
        landmark_x = []
        landmark_y = []
        distance = []
        bearing = []
        if len(sens.readings) > 0:
            for i in range(len(sens.readings)):
                distance_hat = math.sqrt((x_k1_hat[0] - sens.readings[i].landmark.x)**2 + (x_k1_hat[1] - sens.readings[i].landmark.y)**2)
                if distance_hat > 0.1:
                #if sens.readings[i].range > 0.1:
                    landmark_x.append(sens.readings[i].landmark.x)
                    landmark_y.append(sens.readings[i].landmark.y)
                    distance.append(sens.readings[i].range)
                    while sens.readings[i].bearing > math.pi:
                        sens.readings[i].bearing -= 2*math.pi
                    while sens.readings[i].bearing < -math.pi:
                        sens.readings[i].bearing += 2*math.pi
                    bearing.append(sens.readings[i].bearing)
                else:
                    print("deleted")

            
        if len(distance) >= 1:
            w = numpy.zeros((2*len(distance),2*len(distance)))
            for i in range(len(distance)):
                w[2*i,2*i] = 0.1
                w[2*i+1,2*i+1] = 0.05

            H_k1 = numpy.zeros((2*len(distance),3))
            for i in range(len(distance)):
                H_k1[2*i,0] = (x_k1_hat[0] - landmark_x[i])/math.sqrt((x_k1_hat[0] - landmark_x[i])**2 + (x_k1_hat[1] - landmark_y[i])**2)
                H_k1[2*i,1] = (x_k1_hat[1] - landmark_y[i])/math.sqrt((x_k1_hat[0] - landmark_x[i])**2 + (x_k1_hat[1] - landmark_y[i])**2)
                H_k1[2*i,2] = 0
                H_k1[2*i+1,0] = (-x_k1_hat[1] + landmark_y[i])/((x_k1_hat[0] - landmark_x[i])**2 + (x_k1_hat[1] - landmark_y[i])**2)
                H_k1[2*i+1,1] = (x_k1_hat[0] - landmark_x[i])/((x_k1_hat[0] - landmark_x[i])**2 + (x_k1_hat[1] - landmark_y[i])**2)
                H_k1[2*i+1,2] = -1


            S = numpy.dot(numpy.dot(H_k1,P_k1_hat),numpy.transpose(H_k1)) + w
            R = numpy.dot(numpy.dot(P_k1_hat,numpy.transpose(H_k1)),numpy.linalg.inv(S))

            y_hat = numpy.zeros((2*len(distance),1))
            for i in range(len(distance)):
                y_hat[2*i] = math.sqrt((x_k1_hat[0] - landmark_x[i])**2+(x_k1_hat[1] - landmark_y[i])**2)
                y_hat[2*i+1] = math.atan2(-x_k1_hat[1] + landmark_y[i], -x_k1_hat[0] + landmark_x[i]) - x_k1_hat[2]
                while y_hat[2*i+1] > math.pi:
                    y_hat[2*i+1] -= 2*math.pi
                while y_hat[2*i+1] < -math.pi:
                    y_hat[2*i+1] += 2*math.pi

            nu = numpy.zeros((2*len(distance),1))
            for i in range(len(distance)):
                nu[2*i] = distance[i] - y_hat[2*i,0]
                nu[2*i+1] = bearing[i] - y_hat[2*i+1,0]
            for i in range(len(distance)):    
                if nu[2*i+1] < -math.pi:
                    nu[2*i+1] += 2*math.pi
                if nu[2*i+1] > math.pi:
                    nu[2*i+1] -= 2*math.pi

                
                    

            x_k1 = x_k1_hat + numpy.dot(R,nu)
            P_k1 = P_k1_hat - numpy.dot(numpy.dot(R,H_k1),P_k1_hat)

        else:
            x_k1 = x_k1_hat
            P_k1 = P_k1_hat
            

        self.x = x_k1
        self.P = P_k1

        #### ----- YOUR CODE GOES HERE ----- ####
    
    def sensor_callback(self,sens):

        # Publish state estimate 
        self.estimate(sens)
        est_msg = RobotPose()
        est_msg.header.stamp = sens.header.stamp
        est_msg.pose.x = self.x[0]
        est_msg.pose.y = self.x[1]
        est_msg.pose.theta = self.x[2]
        self.pub_est.publish(est_msg)

if __name__ == '__main__':
    rospy.init_node('state_estimator', anonymous=True)
    est = Estimator()
    rospy.spin()
