#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer
from cv_bridge import CvBridge
import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import copy
from PIL import Image
import rospkg 


"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')

        # initialise for image conversion
        self.bridge = CvBridge()

        # find the calibration parameters
        self.camera_info = self.load_intrinsics()
        rospy.loginfo(f'Camera info: {self.camera_info}')
        self.homography = self.load_extrinsics()
        rospy.loginfo(f'Homography: {self.homography}')
        rospy.loginfo("Calibration parameters extracted.")

        # construct publisher to images
        image_pub_topic = f'/{self.veh}/{node_name}/augmented_image/image/compressed'
        self.image_pub = rospy.Publisher(image_pub_topic, CompressedImage, queue_size=16)
        rospy.loginfo(f'Publishing to: {image_pub_topic}')

        # construct subscriber to images
        image_sub_topic = f'/{self.veh}/camera_node/image/compressed'
        self.image_sub = rospy.Subscriber(image_sub_topic, CompressedImage, self.callback)
        rospy.loginfo(f'Subscribed to: {image_sub_topic}')


    def callback(self, image_msg):
        """ Once recieving the image, save the data in the correct format
            is it okay to do a lot of computation in the callback?
        """
        rospy.loginfo('Image recieved, running callback.')

        # extract the image message to a cv image
        image_np = self.readImage(image_msg)

        # detect aprtil tag and extract its reference frame
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) # or BGR?
        print(type(image_gray))
        print(image_gray.dtype)
        # self.detect(image_gray)

        augmented_image = image_gray

        # make new CompressedImage to publish
        augmented_image_msg = CompressedImage()
        augmented_image_msg.header.stamp = rospy.Time.now()
        augmented_image_msg.format = "jpeg"
        augmented_image_msg.data = np.array(cv2.imencode('.jpg', augmented_image)[1]).tostring()

        # Publish new image
        self.image_pub.publish(augmented_image_msg)
        rospy.loginfo('Callback completed, publishing image')


    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        self.frame_id = self.veh + '/camera_optical_frame'
        self.cali_file = cali_file_folder + self.veh + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        rospy.loginfo(f'Looking for calibration {self.cali_file}')
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        calib_data = self.readYamlFile(self.cali_file)
        self.log("Using calibration file: %s" % self.cali_file)

        return calib_data


    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + self.veh + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(f"Can't find calibration file: {cali_file}.\n Using default calibration instead.", 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        calib_data = self.readYamlFile(cali_file)

        return calib_data['homography']


    def detect(self, img, estimate_tag_pose=False, camera_params=None, tag_size=None):

        '''
            Run detectons on the provided image. The image must be a grayscale
            image of type numpy.uint8.
            estimate_tag_pose ?
            camera_params: what format?
            tag_size: 
        '''

        assert len(img.shape) == 2
        assert img.dtype == np.uint8

        c_img = self._convert_image(img)

        return_info = []

        # detect apriltags in the image
        self.libc.apriltag_detector_detect.restype = ctypes.POINTER(_ZArray)
        detections = self.libc.apriltag_detector_detect(self.tag_detector_ptr, c_img)

        apriltag = ctypes.POINTER(_ApriltagDetection)()


        for i in range(0, detections.contents.size):

            #extract the data for each apriltag that was identified
            zarray_get(detections, i, ctypes.byref(apriltag))

            tag = apriltag.contents

            homography = _matd_get_array(tag.H).copy() # numpy.zeros((3,3))  # Don't ask questions, move on with your life
            center = numpy.ctypeslib.as_array(tag.c, shape=(2,)).copy()
            corners = numpy.ctypeslib.as_array(tag.p, shape=(4, 2)).copy()

            detection = Detection()
            detection.tag_family = ctypes.string_at(tag.family.contents.name)
            detection.tag_id = tag.id
            detection.hamming = tag.hamming
            detection.decision_margin = tag.decision_margin
            detection.homography = homography
            detection.center = center
            detection.corners = corners

            if estimate_tag_pose:
                if camera_params==None:
                    raise Exception('camera_params must be provided to detect if estimate_tag_pose is set to True')
                if tag_size==None:
                    raise Exception('tag_size must be provided to detect if estimate_tag_pose is set to True')

                camera_fx, camera_fy, camera_cx, camera_cy = [ c for c in camera_params ]

                info = _ApriltagDetectionInfo(det=apriltag,
                                              tagsize=tag_size,
                                              fx=camera_fx,
                                              fy=camera_fy,
                                              cx=camera_cx,
                                              cy=camera_cy)
                pose = _ApriltagPose()

                self.libc.estimate_tag_pose.restype = ctypes.c_double
                err = self.libc.estimate_tag_pose(ctypes.byref(info), ctypes.byref(pose))

                detection.pose_R = _matd_get_array(pose.R).copy()
                detection.pose_t = _matd_get_array(pose.t).copy()
                detection.pose_err = err


            #Append this dict to the tag data array
            return_info.append(detection)

        self.libc.image_u8_destroy.restype = None
        self.libc.image_u8_destroy(c_img)

        self.libc.apriltag_detections_destroy.restype = None
        self.libc.apriltag_detections_destroy(detections)

        return return_info


    def _convert_image(self, img):

        height = img.shape[0]
        width = img.shape[1]

        self.libc.image_u8_create.restype = ctypes.POINTER(_ImageU8)
        c_img = self.libc.image_u8_create(width, height)

        tmp = _image_u8_get_array(c_img)

        # copy the opencv image into the destination array, accounting for the
        # difference between stride & width.
        tmp[:, :width] = img

        # tmp goes out of scope here but we don't care because
        # the underlying data is still in c_img.
        return c_img

    
    def projection_matrix(self, intrinsic, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """

        #
        # Write your code here
        #


    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []


    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.Loader)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown('No calibration file found.')
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()