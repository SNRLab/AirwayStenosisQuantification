# -*- coding: utf-8 -*-
from __main__ import vtk, qt, ctk, slicer
import slicer # type: ignore
from slicer.ScriptedLoadableModule import * # type: ignore

import os, sys
# Update imports to be relative to module location
moduleDir = os.path.dirname(os.path.realpath(__file__))
if moduleDir not in sys.path:
    sys.path.append(moduleDir)

import numpy as np


class AirwayStenosisQuantification(ScriptedLoadableModule): # type: ignore
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent) # type: ignore
        parent.title = "Airway Stenosis Quantification"
        parent.categories = ["Utilities"]
        parent.contributors = ["Franklin King, Artur Banach"]
        
        parent.helpText = """
        Add help text
        """
        parent.acknowledgementText = """
        """
        self.parent = parent
        
        moduleDir = os.path.dirname(self.parent.path)
        for iconExtension in ['.svg', '.png']:
            iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
            if os.path.isfile(iconPath):
                parent.icon = qt.QIcon(iconPath)
                break


class AirwayStenosisQuantificationWidget(ScriptedLoadableModuleWidget): # type: ignore
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent) # type: ignore
        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
        self.layout = self.parent.layout()
        if not parent:
            self.setup()
            self.parent.show()

    def onReload(self, moduleName="AirwayStenosisQuantification"):
        if 'StenosisUtils.depth_estimation' in sys.modules:
            del sys.modules['StenosisUtils.depth_estimation']

        self._setNodeObserver(None, "bronchVideo")
        self._setNodeObserver(None, "expiration")
        self._setNodeObserver(None, "inspiration")

        for node in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode'):
            if node.GetName().endswith('_Depth'):
                slicer.mrmlScene.RemoveNode(node)

        self.module_initialized = False
        globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self) # type: ignore
        self.logic = AirwayStenosisQuantificationLogic()
        self.initializeDepthEstimation()

        #------------------------------------------------------------------
        #------------------------Startup-----------------------------------
        self.startupButton = ctk.ctkCollapsibleButton()
        self.startupButton.text = "Setup"
        self.layout.addWidget(self.startupButton)

        startupLayout = qt.QFormLayout(self.startupButton)

        self.setupLayoutButton = qt.QPushButton("Set Up Layout")
        startupLayout.addRow(self.setupLayoutButton)
        self.setupLayoutButton.connect('clicked(bool)', self.onSetupLayout)

        #------------------------------------------------------------------
        #------------------------Bronchoscope------------------------------
        self.bronchoscopeButton = ctk.ctkCollapsibleButton()
        self.bronchoscopeButton.text = "Bronchoscope"
        self.layout.addWidget(self.bronchoscopeButton)

        bronchoscopeLayout = qt.QFormLayout(self.bronchoscopeButton)

        self.bronchVideoSelector = slicer.qMRMLNodeComboBox()
        self.bronchVideoSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.bronchVideoSelector.selectNodeUponCreation = False
        self.bronchVideoSelector.addEnabled = False
        self.bronchVideoSelector.removeEnabled = False
        self.bronchVideoSelector.noneEnabled = True
        self.bronchVideoSelector.noneDisplay = "Select Bronchoscope Video..."
        self.bronchVideoSelector.showHidden = False
        self.bronchVideoSelector.setMRMLScene(slicer.mrmlScene)
        self.bronchVideoSelector.setToolTip("Select the live bronchoscope video volume")
        bronchoscopeLayout.addRow("Bronchoscope Video:", self.bronchVideoSelector)

        self.snapshotExpirationButton = qt.QPushButton("Snapshot Expiration Image")
        bronchoscopeLayout.addRow(self.snapshotExpirationButton)
        self.snapshotExpirationButton.connect('clicked(bool)', self.onSnapshotExpiration)

        self.snapshotInspirationButton = qt.QPushButton("Snapshot Inspiration Image")
        bronchoscopeLayout.addRow(self.snapshotInspirationButton)
        self.snapshotInspirationButton.connect('clicked(bool)', self.onSnapshotInspiration)

        #------------------------------------------------------------------
        #------------------------Stenosis Index----------------------------
        self.stenosisIndexButton = ctk.ctkCollapsibleButton()
        self.stenosisIndexButton.text = "Stenosis Index"
        self.layout.addWidget(self.stenosisIndexButton)

        stenosisIndexLayout = qt.QFormLayout(self.stenosisIndexButton)

        self.expirationSelector = slicer.qMRMLNodeComboBox()
        self.expirationSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.expirationSelector.selectNodeUponCreation = False
        self.expirationSelector.addEnabled = False
        self.expirationSelector.removeEnabled = False
        self.expirationSelector.noneEnabled = True
        self.expirationSelector.noneDisplay = "Select Expiration Image..."
        self.expirationSelector.showHidden = False
        self.expirationSelector.setMRMLScene(slicer.mrmlScene)
        self.expirationSelector.setToolTip("Select the expiration image volume")
        stenosisIndexLayout.addRow("Expiration Image:", self.expirationSelector)

        self.inspirationSelector = slicer.qMRMLNodeComboBox()
        self.inspirationSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inspirationSelector.selectNodeUponCreation = False
        self.inspirationSelector.addEnabled = False
        self.inspirationSelector.removeEnabled = False
        self.inspirationSelector.noneEnabled = True
        self.inspirationSelector.noneDisplay = "Select Inspiration Image..."
        self.inspirationSelector.showHidden = False
        self.inspirationSelector.setMRMLScene(slicer.mrmlScene)
        self.inspirationSelector.setToolTip("Select the inspiration image volume")
        stenosisIndexLayout.addRow("Inspiration Image:", self.inspirationSelector)

        self.expirationThresholdSlider = ctk.ctkSliderWidget()
        self.expirationThresholdSlider.minimum = 0
        self.expirationThresholdSlider.maximum = 128
        self.expirationThresholdSlider.value = 35
        self.expirationThresholdSlider.singleStep = 1
        self.expirationThresholdSlider.setToolTip("Threshold for expiration depth map")
        stenosisIndexLayout.addRow("Expiration Threshold:", self.expirationThresholdSlider)

        self.inspirationThresholdSlider = ctk.ctkSliderWidget()
        self.inspirationThresholdSlider.minimum = 0
        self.inspirationThresholdSlider.maximum = 128
        self.inspirationThresholdSlider.value = 20
        self.inspirationThresholdSlider.singleStep = 1
        self.inspirationThresholdSlider.setToolTip("Threshold for inspiration depth map")
        stenosisIndexLayout.addRow("Inspiration Threshold:", self.inspirationThresholdSlider)

        self.expirationThresholdSlider.connect('valueChanged(double)', self.onExpirationThresholdChanged)
        self.inspirationThresholdSlider.connect('valueChanged(double)', self.onInspirationThresholdChanged)

        siDisplayFrame = qt.QFrame()
        siDisplayFrame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 2px solid #555555;
                border-radius: 8px;
                padding: 8px;
                margin-top: 6px;
            }
        """)
        siDisplayLayout = qt.QVBoxLayout(siDisplayFrame)

        siTitleLabel = qt.QLabel("Stenosis Index")
        siTitleLabel.setAlignment(qt.Qt.AlignCenter)
        siTitleLabel.setStyleSheet("QLabel { color: #aaaaaa; font-size: 11px; font-weight: bold; border: none; background: transparent; }")
        siDisplayLayout.addWidget(siTitleLabel)

        self.siValueLabel = qt.QLabel("—")
        self.siValueLabel.setAlignment(qt.Qt.AlignCenter)
        self.siValueLabel.setStyleSheet("QLabel { color: #4FC3F7; font-size: 36px; font-weight: bold; border: none; background: transparent; }")
        siDisplayLayout.addWidget(self.siValueLabel)

        self.siDetailLabel = qt.QLabel("")
        self.siDetailLabel.setAlignment(qt.Qt.AlignCenter)
        self.siDetailLabel.setStyleSheet("QLabel { color: #888888; font-size: 10px; border: none; background: transparent; }")
        siDisplayLayout.addWidget(self.siDetailLabel)

        stenosisIndexLayout.addRow(siDisplayFrame)

        self.layout.addStretch(1)

        self.bronchVideoSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onBronchVideoNodeChanged)
        self.expirationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onExpirationNodeChanged)
        self.inspirationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInspirationNodeChanged)

        self.bronchVideoObserverTag = None
        self.bronchVideoObserverNode = None
        self.expirationObserverTag = None
        self.expirationObserverNode = None
        self.inspirationObserverTag = None
        self.inspirationObserverNode = None

    def onBronchVideoNodeChanged(self, node):
        self._setNodeObserver(node, "bronchVideo")
        self.runBronchDepthEstimation(node)

    def runBronchDepthEstimation(self, node):
        if node is None:
            return
        try:
            from PIL import Image
            slicer.app.processEvents()

            input_array = slicer.util.arrayFromVolume(node)
            if input_array.ndim == 4:
                input_array = np.mean(input_array[..., :3], axis=-1).astype(np.uint8)
            if input_array.ndim == 3:
                input_array = input_array[0]
            pil_image = Image.fromarray(input_array.astype(np.uint8), mode='L')

            depth_image = self.logic.depth_estimator.generateDepth(pil_image)
            output_name = node.GetName() + "_Depth"
            self.displayPILImageInSlicer(depth_image, output_name)
        except Exception as e:
            pass

    def onSnapshotExpiration(self):
        self.snapshotToSelector(self.expirationSelector)

    def onSnapshotInspiration(self):
        self.snapshotToSelector(self.inspirationSelector)

    def snapshotToSelector(self, selector):
        sourceNode = self.bronchVideoSelector.currentNode()
        targetNode = selector.currentNode()
        if sourceNode is None or targetNode is None:
            return

        source_array = slicer.util.arrayFromVolume(sourceNode)
        if source_array.ndim == 3 and source_array.shape[0] == 1:
            gray_array = source_array
        elif source_array.ndim == 4:
            gray_array = np.mean(source_array[..., :3], axis=-1).astype(np.uint8)
            if gray_array.ndim == 2:
                gray_array = gray_array[np.newaxis, ...]
        elif source_array.ndim == 3 and source_array.shape[0] > 1:
            gray_array = np.mean(source_array, axis=0, keepdims=True).astype(np.uint8)
        else:
            gray_array = source_array.copy()

        targetNode.SetIJKToRASDirections(1,0,0, 0,1,0, 0,0,1)
        slicer.util.updateVolumeFromArray(targetNode, gray_array)

    def onSetupLayout(self):
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourByTwoSliceView)

        bronchNode = self.bronchVideoSelector.currentNode()
        expirationNode = self.expirationSelector.currentNode()
        inspirationNode = self.inspirationSelector.currentNode()

        sliceAssignments = {}
        if bronchNode:
            bronchDepthName = bronchNode.GetName() + "_Depth"
            bronchDepthNode = slicer.util.getFirstNodeByName(bronchDepthName)
            sliceAssignments["Red"] = bronchNode
            if bronchDepthNode:
                sliceAssignments["Slice5"] = bronchDepthNode
        if expirationNode:
            expirationDepthName = expirationNode.GetName() + "_Depth"
            expirationDepthNode = slicer.util.getFirstNodeByName(expirationDepthName)
            expirationThresholdName = expirationNode.GetName() + "_Threshold"
            expirationThresholdNode = slicer.util.getFirstNodeByName(expirationThresholdName)
            sliceAssignments["Yellow"] = expirationNode
            if expirationDepthNode:
                sliceAssignments["Green"] = expirationDepthNode
            if expirationThresholdNode:
                sliceAssignments["Slice4"] = expirationThresholdNode
        if inspirationNode:
            inspirationDepthName = inspirationNode.GetName() + "_Depth"
            inspirationDepthNode = slicer.util.getFirstNodeByName(inspirationDepthName)
            inspirationThresholdName = inspirationNode.GetName() + "_Threshold"
            inspirationThresholdNode = slicer.util.getFirstNodeByName(inspirationThresholdName)
            sliceAssignments["Slice6"] = inspirationNode
            if inspirationDepthNode:
                sliceAssignments["Slice7"] = inspirationDepthNode
            if inspirationThresholdNode:
                sliceAssignments["Slice8"] = inspirationThresholdNode

        foregroundOverlays = {}
        if expirationNode and expirationThresholdNode:
            foregroundOverlays["Yellow"] = expirationThresholdNode
        if inspirationNode and inspirationThresholdNode:
            foregroundOverlays["Slice6"] = inspirationThresholdNode

        for sliceName, volumeNode in sliceAssignments.items():
            sliceWidget = layoutManager.sliceWidget(sliceName)
            if sliceWidget is None:
                continue
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(volumeNode.GetID())
            if sliceName in foregroundOverlays:
                compositeNode.SetForegroundVolumeID(foregroundOverlays[sliceName].GetID())
                compositeNode.SetForegroundOpacity(0.3)
            sliceWidget.mrmlSliceNode().SetOrientationToAxial()
            sliceWidget.sliceLogic().FitSliceToAll()

    def onExpirationThresholdChanged(self, value):
        self.updateThresholdImage("expiration")
        self.updateStenosisIndex()

    def onInspirationThresholdChanged(self, value):
        self.updateThresholdImage("inspiration")
        self.updateStenosisIndex()

    def updateThresholdImage(self, which):
        if which == "expiration":
            sourceNode = self.expirationSelector.currentNode()
            threshold = self.expirationThresholdSlider.value
        else:
            sourceNode = self.inspirationSelector.currentNode()
            threshold = self.inspirationThresholdSlider.value

        if sourceNode is None:
            return

        depthName = sourceNode.GetName() + "_Depth"
        depthNode = slicer.util.getFirstNodeByName(depthName)
        if depthNode is None:
            return

        depth_array = slicer.util.arrayFromVolume(depthNode)
        thresholded = np.where(depth_array > threshold * 2, 255, 0).astype(np.uint8)

        outputName = sourceNode.GetName() + "_Threshold"
        self.displayArrayInSlicer(thresholded, outputName)

    def updateStenosisIndex(self):
        expirationNode = self.expirationSelector.currentNode()
        inspirationNode = self.inspirationSelector.currentNode()
        if expirationNode is None or inspirationNode is None:
            self.siValueLabel.setText("—")
            self.siDetailLabel.setText("Select both images to calculate")
            return

        expDepthNode = slicer.util.getFirstNodeByName(expirationNode.GetName() + "_Depth")
        inspDepthNode = slicer.util.getFirstNodeByName(inspirationNode.GetName() + "_Depth")
        if expDepthNode is None or inspDepthNode is None:
            self.siValueLabel.setText("—")
            self.siDetailLabel.setText("Waiting for depth maps...")
            return

        expThreshold = self.expirationThresholdSlider.value * 2
        inspThreshold = self.inspirationThresholdSlider.value * 2

        exp_array = slicer.util.arrayFromVolume(expDepthNode)
        insp_array = slicer.util.arrayFromVolume(inspDepthNode)

        exp_count = int(np.sum(exp_array > expThreshold))
        insp_count = int(np.sum(insp_array > inspThreshold))

        if insp_count == 0:
            self.siValueLabel.setText("N/A")
            self.siDetailLabel.setText("Inspiration pixel count is zero")
            self.siValueLabel.setStyleSheet("QLabel { color: #CD5C5C; font-size: 36px; font-weight: bold; border: none; background: transparent; }")
            return

        si = 1.0 - (exp_count / insp_count)

        if si < 0.25:
            color = "#4FC3F7"
        elif si < 0.50:
            color = "#FFF176"
        elif si < 0.75:
            color = "#FFB74D"
        else:
            color = "#EF5350"

        self.siValueLabel.setText(f"{si:.4f}")
        self.siValueLabel.setStyleSheet(f"QLabel {{ color: {color}; font-size: 36px; font-weight: bold; border: none; background: transparent; }}")
        self.siDetailLabel.setText(f"Exp: {exp_count} px  |  Insp: {insp_count} px")

    def onExpirationNodeChanged(self, node):
        if node is not None:
            node.SetIJKToRASDirections(1,0,0, 0,1,0, 0,0,1)
        self._setNodeObserver(node, "expiration")
        self.runDepthEstimation(node)

    def onInspirationNodeChanged(self, node):
        if node is not None:
            node.SetIJKToRASDirections(1,0,0, 0,1,0, 0,0,1)
        self._setNodeObserver(node, "inspiration")
        self.runDepthEstimation(node)

    def _setNodeObserver(self, node, slot):
        old_node = getattr(self, f"{slot}ObserverNode")
        old_tag = getattr(self, f"{slot}ObserverTag")
        if old_node and old_tag:
            old_node.RemoveObserver(old_tag)

        if node is not None:
            callback = self.runBronchDepthEstimation if slot == "bronchVideo" else self.runDepthEstimation
            tag = node.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent,
                                   lambda caller, event, n=node: callback(n))
            setattr(self, f"{slot}ObserverNode", node)
            setattr(self, f"{slot}ObserverTag", tag)
        else:
            setattr(self, f"{slot}ObserverNode", None)
            setattr(self, f"{slot}ObserverTag", None)

    def initializeDepthEstimation(self):
        from StenosisUtils.depth_estimation import DepthEstimator
        self.logic.depth_estimator = DepthEstimator()
        print("Depth Estimation Model Initialized")

    def runDepthEstimation(self, node):
        if node is None:
            return
        try:
            slicer.app.processEvents()
            depth_node = self.generateDepthMap(node)
            expirationNode = self.expirationSelector.currentNode()
            inspirationNode = self.inspirationSelector.currentNode()
            if expirationNode and node.GetID() == expirationNode.GetID():
                self.updateThresholdImage("expiration")
            if inspirationNode and node.GetID() == inspirationNode.GetID():
                self.updateThresholdImage("inspiration")
            self.updateStenosisIndex()
        except Exception as e:
            pass

    def generateDepthMap(self, inputVolumeNode):
        """Generate a depth map from a vtkMRMLScalarVolumeNode.

        Args:
            inputVolumeNode: vtkMRMLScalarVolumeNode containing the input image.

        Returns:
            vtkMRMLScalarVolumeNode containing the depth map.
        """
        from PIL import Image

        input_array = slicer.util.arrayFromVolume(inputVolumeNode)
        if input_array.ndim == 4:
            input_array = np.mean(input_array[..., :3], axis=-1).astype(np.uint8)
        if input_array.ndim == 3:
            input_array = input_array[0]
        pil_image = Image.fromarray(input_array.astype(np.uint8), mode='L')

        depth_image = self.logic.depth_estimator.generateDepth(pil_image)

        output_name = inputVolumeNode.GetName() + "_Depth"
        depth_node = self.displayPILImageInSlicer(depth_image, output_name)
        return depth_node

    def displayPILImageInSlicer(self, pil_image, node_name):
        """Display a PIL image as a vtkMRMLScalarVolumeNode.

        Args:
            pil_image: PIL Image to display.
            node_name: Name for the Slicer volume node.

        Returns:
            vtkMRMLScalarVolumeNode with the image data.
        """
        image_array = np.array(pil_image)
        return self.displayArrayInSlicer(image_array, node_name)

    def displayArrayInSlicer(self, image_array, node_name):
        """Display a numpy array as a vtkMRMLScalarVolumeNode.

        Args:
            image_array: 2-D or 3-D numpy array.
            node_name: Name for the Slicer volume node.

        Returns:
            vtkMRMLScalarVolumeNode with the image data.
        """
        if len(image_array.shape) == 2:
            image_array = image_array[np.newaxis, ...]

        node = slicer.util.getFirstNodeByName(node_name)
        if node is None:
            node = slicer.vtkMRMLScalarVolumeNode()
            node.SetName(node_name)
            slicer.mrmlScene.AddNode(node)

        node.SetIJKToRASDirections(1,0,0, 0,1,0, 0,0,1)
        slicer.util.updateVolumeFromArray(node, image_array)
        return node


class AirwayStenosisQuantificationLogic(ScriptedLoadableModuleLogic): # type: ignore
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self) # type: ignore
        self.depth_estimator = None
