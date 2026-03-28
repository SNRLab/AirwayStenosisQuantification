import logging
import os

import vtk
import subprocess
import time
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import cv2
import numpy as np

#import depth_estimation
#os.system('/bin/bash  --rcfile /venv/Scripts/activate')
#import depth_estimation


#import the depth estimation
'''import importlib.util
import sys
module = importlib.util.spec_from_file_location("module.name", "/path/to/file.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)
foo.MyClass()'''

#depth estimation imports
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import cv2
        #import sklearn

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
print(torch.__version__)                



#depth estimation intro
parser = argparse.ArgumentParser()
parser.add_argument("-network_name", type=str, default="6Level", help="name of the network")

parser.add_argument("--dataset_name", type=str, default="#1245", help="name of the training dataset")
parser.add_argument("--testing_dataset", type=str, default="AS", help="name of the testing dataset")
parser.add_argument("--lambda_cyc", type=float, default=0.1, help="cycle loss weight")

parser.add_argument("--epoch", type=int, default=7, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=200, help="size of image height")
parser.add_argument("--img_width", type=int, default=200, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--textfile_training_results_interval", type=int, default=50,
                    help="textfile_training_results_interval")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_id", type=float, default=1, help="identity loss weight")
opt = parser.parse_args()
print(opt)


cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.img_height, opt.img_width)
# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
  G_AB = G_AB.cuda()
  

if opt.epoch != 0:
  # Load pretrained models
  module_dir = os.path.dirname(__file__)
  G_AB.load_state_dict(torch.load(os.path.join(module_dir, "6Level-ex-vivo-G_AB.pth")))


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_B1_buffer = ReplayBuffer()

        





#
# Airway_Stenosis
#

class Airway_Stenosis(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Airway_Stenosis"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Airway_Stenosis">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Airway_Stenosis1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Airway_Stenosis',
        sampleName='Airway_Stenosis1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'Airway_Stenosis1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='Airway_Stenosis1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='Airway_Stenosis1'
    )

    # Airway_Stenosis2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='Airway_Stenosis',
        sampleName='Airway_Stenosis2',
        thumbnailFileName=os.path.join(iconsPath, 'Airway_Stenosis2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='Airway_Stenosis2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='Airway_Stenosis2'
    )


#
# Airway_StenosisWidget
#

class Airway_StenosisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Airway_Stenosis.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = Airway_StenosisLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.ExpirationEdge.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.ExpirationThresholdSlider.connect('valueChanged(double)', self.onThresholdSlider)
        

        # Buttons
        self.ui.showImagesButton.connect('clicked(bool)', self.onShowImagesButton)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.SegmentCTButton.connect('clicked(bool)', self.onSegmentCTButton)
        self.ui.CalcGTSIButton.connect('clicked(bool)', self.onCalcGTSIButton)
        self.ui.UpdateDirsButton.connect('clicked(bool)',self.onUpdateDirsButton)

        #Initialize the paths
        self.ui.MyPCPath.currentPath = ""# 
        self.ui.ExpirationPath.currentPath = ""  
        self.ui.InspirationPath.currentPath = ""  
        #Markups
        self.markupsLogic = slicer.modules.markups.logic()
        self.markupsNode1 = slicer.mrmlScene.GetNodeByID(self.markupsLogic.AddNewFiducialNode())
        self.markupsNode1.SetName('Expiration_Markup')

        self.ui.ExpirationEdgeMarker.buttonsVisible = False
        self.ui.ExpirationEdgeMarker.placeButton().show()
        self.ui.ExpirationEdgeMarker.interactionNode()
        self.ui.ExpirationEdgeMarker.setMRMLScene(slicer.mrmlScene)
        self.ui.ExpirationEdgeMarker.setCurrentNode(self.markupsNode1)

        # Set the layout----------------------------------------------------------------------------------------------------------------
        """Set view arrangement, wchich specifies what kind of views are rendered, and their location and sizes.

        :param layoutName: String that specifies the layout name. Most commonly used layouts are:
        `FourUp`, `Conventional`, `OneUp3D`, `OneUpRedSlice`, `OneUpYellowSlice`,
        `OneUpPlot`, `OneUpGreenSlice`, `Dual3D`, `FourOverFour`, `DicomBrowser`.

         Get full list of layout names::

          for att in dir(slicer.vtkMRMLLayoutNode):
               if att.startswith("SlicerLayout"):
                    print(att[12:-4])

        """
        layoutName = "TwoOverTwo"
        layoutId = eval("slicer.vtkMRMLLayoutNode.SlicerLayout"+layoutName+"View")
        slicer.app.layoutManager().setLayout(layoutId)

        #Cleaning the scene from the previous nodes
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("0"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("1"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("depth_0"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("depth_1"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("Expiration_Markup"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("expiration_thresholded"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("inspiration_thresholded"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTableNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTableNode"))

        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("0"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("1"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("depth_0"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("depth_1"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("Expiration_Markup"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("expiration_thresholded"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName("inspiration_thresholded"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTableNode"))
        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTableNode"))
        
        
        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    def onUpdateDirsButton(self):
        #self.ui.ExpirationPath.currentPath = self.ui.MyPCPath.currentPath + "images/input_images/A/0.jpg" #"C:/Users/banac/OneDrive/Desktop/Projects/Bronchoscopy/Stenosis/Airway_Stenosis/Airway_Stenosis/data/Testing/old_gt/AS/A/E4_grey.png"
        #self.ui.InspirationPath.currentPath = self.ui.MyPCPath.currentPath + "/images/input_images/A/1.jpg" #"C:/Users/banac/OneDrive/Desktop/Projects/Bronchoscopy/Stenosis/Airway_Stenosis/Airway_Stenosis/data/Testing/old_gt/AS/A/I4_grey.png"
        print("Masa is boss")
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.ExpirationEdge.setCurrentNode(self._parameterNode.GetNodeReference("ExpirationEdge"))

        # Update buttons states and tooltips
        ''' if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.applyButton.toolTip = "Compute Stenosis Index"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select the Fiducials"
            self.ui.applyButton.enabled = False'''

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("ExpirationEdge", self.ui.ExpirationEdge.currentNodeID)
        

        self._parameterNode.EndModify(wasModified)

    def onShowImagesButton(self):
        """ Displays bronchoscopic images and the displayed depths """
        

        #set 2 by two view
        layoutName = "TwoOverTwo"
        layoutId = eval("slicer.vtkMRMLLayoutNode.SlicerLayout"+layoutName+"View")
        slicer.app.layoutManager().setLayout(layoutId)

        # Create Nodes with images at the chosen location - that displays it in the sce
        self.ExpirationNode = slicer.util.loadVolume(self.ui.ExpirationPath.currentPath,{"singleFile": True})
        self.InspirationNode = slicer.util.loadVolume(self.ui.InspirationPath.currentPath,{"singleFile": True})

        # load the images to python 
        expiration_image = cv2.imread(self.ui.ExpirationPath.currentPath,0)
        inspiration_image = cv2.imread(self.ui.InspirationPath.currentPath,0)

        #save the image to the A folder for depth estimation
        cv2.imwrite(self.ui.MyPCPath.currentPath+"images/input_images/A/0.png", expiration_image)
        cv2.imwrite(self.ui.MyPCPath.currentPath+"images/input_images/A/1.png", inspiration_image)


        #Estimate the depths
        
        self.depth_estimation_testing("lklk")
        

        # Upload the depths into numpy
        self.expiration_depth = cv2.imread(self.ui.MyPCPath.currentPath+"images/estimated_depths/depth0.png",0)
        self.inspiration_depth = cv2.imread(self.ui.MyPCPath.currentPath+"images/estimated_depths/depth1.png",0)

        self.ExpirationNodeDepth = slicer.util.loadVolume(self.ui.MyPCPath.currentPath+'images/estimated_depths/depth0.png',{"singleFile": True})
        self.InspirationNodeDepth = slicer.util.loadVolume(self.ui.MyPCPath.currentPath+'images/estimated_depths/depth1.png',{"singleFile": True})


        # display the right nodes in the right views
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.ExpirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.ExpirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)
        
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.InspirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.InspirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)

        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.ExpirationNodeDepth.GetID())
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.ExpirationNodeDepth.GetID())
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)

        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.InspirationNodeDepth.GetID())
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.InspirationNodeDepth.GetID())
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)


        # Setting the orientation of all windows to Axial---------------------------------------------------------------------------------
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceNode().SetOrientationToAxial()

        # reset FOV
        # reset field of view
        slicer.app.layoutManager().sliceWidget('Red').fitSliceToBackground()
        slicer.app.layoutManager().sliceWidget('Green').fitSliceToBackground()
        slicer.app.layoutManager().sliceWidget('Slice4').fitSliceToBackground()
        slicer.app.layoutManager().sliceWidget('Yellow').fitSliceToBackground()

    def onApplyButton(self):

        #get the depth edge from the marker 10 pixels arund the point
        pos_RAS = [0,0,0]
        self.ui.ExpirationEdge.currentNode().GetNthFiducialPosition(0,pos_RAS) 	
        

        #we need to map the coordinates of the fiducials to the coordinates in the photo
        x=int((-1)*pos_RAS[0])
        y=int((-1)*pos_RAS[1])

        threshold = 0
        for i in range (0,10):
            for j in range (0,10):
                if self.expiration_depth[x+i,y+j]/2 > threshold:
                    threshold = self.expiration_depth[x+i,y+j]/2
                if self.expiration_depth[x-i,y-j]/2 > threshold:
                    threshold = self.expiration_depth[x-i,y-j]/2
        print(threshold)        


        # Threshold them 
        nan_nr_expiration=0
        nan_nr_inspiration=0
        expiration_depth_thresholded = np.zeros((200,200))
        inspiration_depth_thresholded = np.zeros((200,200))

        self.ui.ExpirationThresholdSlider.value = threshold


        for i in range (0,200):
            for j in range (0,200):
                if self.expiration_depth[i,j] > threshold*2:
                    expiration_depth_thresholded[i,j]= 255
                    nan_nr_expiration= nan_nr_expiration+1
                
                if self.inspiration_depth[i,j] > threshold*2:
                    inspiration_depth_thresholded[i,j]= 255
                    nan_nr_inspiration= nan_nr_inspiration+1

        

        #save them to files
        cv2.imwrite(self.ui.MyPCPath.currentPath + 'images/expiration_thresholded.jpg', expiration_depth_thresholded)
        cv2.imwrite(self.ui.MyPCPath.currentPath+ 'images/inspiration_thresholded.jpg', inspiration_depth_thresholded)

        #Add them as nodes to display
        self.ExpirationNodeThresholded = slicer.util.loadVolume(self.ui.MyPCPath.currentPath+'images/expiration_thresholded.jpg')
        self.InspirationNodeThresholded = slicer.util.loadVolume(self.ui.MyPCPath.currentPath+'images/inspiration_thresholded.jpg')

        # Do the maths to calculate SI
        SI = 1-(nan_nr_expiration/nan_nr_inspiration)
        self.ui.lcdNumber.display(SI)
        print("SI: ", SI)


        # display the right nodes in the right views
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.ExpirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(1)

        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.InspirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(1)

        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.ExpirationNodeThresholded.GetID())
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)

        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.InspirationNodeThresholded.GetID())
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)


        # Setting the orientation of all windows to Axial---------------------------------------------------------------------------------
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceNode().SetOrientationToAxial()

        # reset FOV
        slicer.app.layoutManager().sliceWidget('Green').fitSliceToBackground()
        slicer.app.layoutManager().sliceWidget('Yellow').fitSliceToBackground()

    def onThresholdSlider(self,threshold):

        # Threshold them 
        SI = 0
        nan_nr_expiration=0
        nan_nr_inspiration=0
        expiration_depth_thresholded = np.zeros((200,200))
        inspiration_depth_thresholded = np.zeros((200,200))

        for i in range (0,200):
            for j in range (0,200):
                if self.expiration_depth[i,j] > self.ui.ExpirationThresholdSlider.value*2:
                    expiration_depth_thresholded[i,j]= 255
                    nan_nr_expiration= nan_nr_expiration+1
                if self.inspiration_depth[i,j] > self.ui.ExpirationThresholdSlider.value*2:
                    inspiration_depth_thresholded[i,j]= 255
                    nan_nr_inspiration= nan_nr_inspiration+1

        #save them to files
        cv2.imwrite(self.ui.MyPCPath.currentPath+'images/expiration_thresholded.jpg', expiration_depth_thresholded)
        cv2.imwrite(self.ui.MyPCPath.currentPath+'images/inspiration_thresholded.jpg', inspiration_depth_thresholded)

        #Add them as nodes to display
        self.ExpirationNodeThresholded = slicer.util.loadVolume(self.ui.MyPCPath.currentPath+'images/expiration_thresholded.jpg')
        self.InspirationNodeThresholded = slicer.util.loadVolume(self.ui.MyPCPath.currentPath+'images/inspiration_thresholded.jpg')

        # Do the maths to calculate SI
        SI = 1-(nan_nr_expiration/nan_nr_inspiration)
        self.ui.lcdNumber.display(SI)
        print("SI: ", SI)


        # display the right nodes in the right views
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.ExpirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(1)

        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.InspirationNode.GetID())
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(1)

        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.ExpirationNodeThresholded.GetID())
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)

        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.InspirationNodeThresholded.GetID())
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0)


        # Setting the orientation of all windows to Axial---------------------------------------------------------------------------------
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceNode().SetOrientationToAxial()
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceNode().SetOrientationToAxial()

        # reset FOV
        slicer.app.layoutManager().sliceWidget('Green').sliceLogic().GetSliceNode().SetFieldOfView(401.86915887850466, 200.0, 1.0)
        slicer.app.layoutManager().sliceWidget('Yellow').sliceLogic().GetSliceNode().SetFieldOfView(401.86915887850466, 200.0, 1.0)

    def onSegmentCTButton(self): 
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.ui.ExpirationCT.currentNode().GetID())
        slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(1)

        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(self.ui.InspirationCT.currentNode().GetID())
        slicer.app.layoutManager().sliceWidget('Slice4').sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(1)


        #Create Segmentations and Segments
        self.exp_segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.exp_segmentationNode.CreateDefaultDisplayNodes()
        self.exp_segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.ui.ExpirationCT.currentNode())# too early!
        self.exp_segmentationNode.SetName("exp_segmentation")
        self.exp_segmentationNode.GetSegmentation().AddEmptySegment("exp_segment")

        self.insp_segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.insp_segmentationNode.CreateDefaultDisplayNodes()
        self.insp_segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.ui.ExpirationCT.currentNode())# too early!
        self.insp_segmentationNode.SetName("insp_segmentation")
        self.insp_segmentationNode.GetSegmentation().AddEmptySegment("insp_segment")


    def onCalcGTSIButton(self):
        #Expiration ------------------------------------------------
        exp_segm  =  slicer.util.getNode('exp_segmentation')        
        expTableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode')
        expTableNode.SetName('Expiration_Table')
     
        import SegmentStatistics
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", exp_segm.GetID())
        segStatLogic.getParameterNode().SetParameter("ScalarVolume", self.ui.ExpirationCT.currentNode().GetID())
        

        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled","False")
        segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.voxel_count.enabled","True")
        segStatLogic.computeStatistics()
        segStatLogic.exportToTable(expTableNode)
        #segStatLogic.showTable(expTableNode)
        #exp_pixels_str = expTableNode.GetCellText(0,1)
        #exp_pixels = int(exp_pixels_str)
        exp_volume_str = expTableNode.GetCellText(0,2)
        exp_volume = int(float(exp_volume_str))
        #print(exp_volume)

        #Inspiration-----------------------------------------------
        insp_segm  =  slicer.util.getNode('insp_segmentation')
        inspTableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode')
        inspTableNode.SetName('Inspiration_Table')
     
        segStatLogic.getParameterNode().SetParameter("Segmentation", insp_segm.GetID())
        segStatLogic.getParameterNode().SetParameter("ScalarVolume", self.ui.InspirationCT.currentNode().GetID())

        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled","False")
        segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.voxel_count.enabled","True")
        segStatLogic.computeStatistics()
        segStatLogic.exportToTable(inspTableNode)
        #segStatLogic.showTable(resultsTableNode)
        #insp_pixels_str = resultsTableNode.GetCellText(0,1)
        #insp_pixels = int(insp_pixels_str)
        insp_volume_str = inspTableNode.GetCellText(0,2)
        insp_volume = int(float(insp_volume_str))
        #print(insp_volume)


        self.ui.GTSI.display(1-(exp_volume/insp_volume))

        

    def depth_estimation_testing(self,bronch_im_path):
        print("sfdagljkh")
        #------------------------------------------------------------------------------------------------------------------------------------------
        transforms_testing_non_fliped_ = [
            # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
            # transforms.RandomCrop((opt.img_height, opt.img_width)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        # Test data loader - non flipped
        val_dataloader_non_flipped = DataLoader(
            ImageDataset(self.ui.MyPCPath.currentPath+"images/input_images", transforms_=transforms_testing_non_fliped_,
                         unaligned=False),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        G_AB.eval()


        for i, batch in enumerate(val_dataloader_non_flipped):
            start = time.time()
            real_A = Variable(batch["A"].type(Tensor))
            fake_B1 = G_AB(real_A)
            end = time.time()
            save_image(fake_B1, self.ui.MyPCPath.currentPath+"images/estimated_depths/depth%s.png" % (i),normalize=False, scale_each=False) #range= (0,128)
       
# Airway_StenosisLogic
#

class Airway_StenosisLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# Airway_StenosisTest
#

class Airway_StenosisTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_Airway_Stenosis1()

    def test_Airway_Stenosis1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('Airway_Stenosis1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = Airway_StenosisLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
