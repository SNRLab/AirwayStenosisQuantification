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
from StenosisUtils.image_utils import toGrayscale2D


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
        for mod in ('StenosisUtils.depth_estimation', 'StenosisUtils.image_utils'):
            if mod in sys.modules:
                del sys.modules[mod]

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
        self.expirationSelector.addEnabled = True
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
        self.inspirationSelector.addEnabled = True
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

            input_array = toGrayscale2D(slicer.util.arrayFromVolume(node))
            pil_image = Image.fromarray(input_array, mode='L')

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
        if source_array.ndim == 3 and 1 not in source_array.shape:
            # Multi-slice scalar volume: average slices
            gray_array = np.mean(source_array, axis=0).astype(np.uint8)
        else:
            gray_array = toGrayscale2D(source_array)
        gray_array = gray_array[np.newaxis, ...]

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

        input_array = toGrayscale2D(slicer.util.arrayFromVolume(inputVolumeNode))
        pil_image = Image.fromarray(input_array, mode='L')

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


_retainedTestWidgets = []


class _IdentityDepthEstimator:
    """Test stand-in for DepthEstimator: returns the input image as the depth map,
    so the depth map content (and thus thresholds / SI) is fully controlled by the test."""
    def __init__(self):
        self.calls = 0

    def generateDepth(self, image):
        self.calls += 1
        return image.copy()


class AirwayStenosisQuantificationTest(ScriptedLoadableModuleTest): # type: ignore
    """Run from Slicer via the module's "Reload and Test" button, or headless with:
        Slicer --no-splash --testing --additional-module-paths <this dir>
               --python-code "slicer.selfTests['AirwayStenosisQuantification']()"
    """

    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.widget = None

    def tearDown(self):
        slicer.mrmlScene.Clear(0)
        if self.widget is not None:
            for slot in ("bronchVideo", "expiration", "inspiration"):
                self.widget._setNodeObserver(None, slot)
            # Deliberately keep test widgets alive: garbage-collecting a
            # ScriptedLoadableModuleWidget whose Qt signals are connected to its
            # bound methods makes Slicer crash on exit.
            _retainedTestWidgets.append(self.widget)
            self.widget = None

    def runTest(self):
        tests = [
            self.test_DisplayArrayCreatesAndReusesNode,
            self.test_DepthMapGeneratedOnSelection,
            self.test_ThresholdImage,
            self.test_StenosisIndex,
            self.test_StenosisIndexMissingInputs,
            self.test_StenosisIndexZeroInspiration,
            self.test_SnapshotConvertsRGBToGrayscale,
            self.test_TrailingSingletonVolumeLayout,
            self.test_BronchVideoDepthUpdatesOnModify,
            self.test_SetupLayout,
            self.test_RealDepthEstimator,
        ]
        for test in tests:
            self.setUp()
            try:
                test()
            finally:
                self.tearDown()
        self.delayDisplay("All AirwayStenosisQuantification tests passed")

    # ------------------------------------------------------------------ helpers

    def _createWidget(self, depthEstimator=None):
        parent = slicer.qMRMLWidget()
        parent.setLayout(qt.QVBoxLayout())
        parent.setMRMLScene(slicer.mrmlScene)
        widget = AirwayStenosisQuantificationWidget(parent)
        # Skip loading the real cGAN; tests inject their own estimator.
        widget.initializeDepthEstimation = lambda: None
        widget.setup()
        widget.logic.depth_estimator = depthEstimator or _IdentityDepthEstimator()
        self.widget = widget
        return widget

    @staticmethod
    def _imageWithBrightPixels(count, value=200, shape=(50, 50)):
        arr = np.zeros(shape, dtype=np.uint8)
        arr.flat[:count] = value
        return arr

    @staticmethod
    def _addVolume(array2d, name):
        return slicer.util.addVolumeFromArray(array2d[np.newaxis, ...], name=name)

    @staticmethod
    def _array(name):
        node = slicer.util.getFirstNodeByName(name)
        return None if node is None else slicer.util.arrayFromVolume(node)

    # ------------------------------------------------------------------ tests

    def test_DisplayArrayCreatesAndReusesNode(self):
        self.delayDisplay("displayArrayInSlicer creates then reuses a node")
        w = self._createWidget()
        first = w.displayArrayInSlicer(np.full((10, 20), 7, np.uint8), "Out")
        self.assertEqual(slicer.util.arrayFromVolume(first).shape, (1, 10, 20))
        second = w.displayArrayInSlicer(np.full((10, 20), 9, np.uint8), "Out")
        self.assertEqual(first.GetID(), second.GetID())
        self.assertEqual(len(slicer.util.getNodes("Out*")), 1)
        self.assertTrue(np.all(slicer.util.arrayFromVolume(second) == 9))

    def test_DepthMapGeneratedOnSelection(self):
        self.delayDisplay("Selecting an expiration image creates _Depth and _Threshold")
        w = self._createWidget()
        img = self._imageWithBrightPixels(100)
        node = self._addVolume(img, "Exp")
        w.expirationSelector.setCurrentNode(node)
        depth = self._array("Exp_Depth")
        self.assertIsNotNone(depth)
        np.testing.assert_array_equal(depth[0], img)
        self.assertIsNotNone(self._array("Exp_Threshold"))

    def test_ThresholdImage(self):
        self.delayDisplay("Threshold image is 255 where depth > 2 * slider value")
        w = self._createWidget()
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        w.inspirationSelector.setCurrentNode(self._addVolume(img, "Insp"))
        for sliderValue in (0, 20, 64, 128):
            w.inspirationThresholdSlider.value = sliderValue
            thresh = self._array("Insp_Threshold")[0]
            expected = np.where(img > sliderValue * 2, 255, 0)
            np.testing.assert_array_equal(thresh, expected)

    def test_StenosisIndex(self):
        self.delayDisplay("Stenosis index = 1 - exp_count / insp_count")
        w = self._createWidget()
        w.expirationSelector.setCurrentNode(self._addVolume(self._imageWithBrightPixels(100), "Exp"))
        w.inspirationSelector.setCurrentNode(self._addVolume(self._imageWithBrightPixels(400), "Insp"))
        self.assertEqual(w.siValueLabel.text, "0.7500")
        self.assertEqual(w.siDetailLabel.text, "Exp: 100 px  |  Insp: 400 px")
        self.assertIn("#EF5350", w.siValueLabel.styleSheet)

        # Raising the expiration threshold above every pixel -> no expiration area -> SI = 1
        w.expirationThresholdSlider.value = 128
        self.assertEqual(w.siValueLabel.text, "1.0000")

        # Equal areas -> SI = 0 (lowest severity color)
        w.expirationThresholdSlider.value = 35
        slicer.util.updateVolumeFromArray(w.expirationSelector.currentNode(),
                                          self._imageWithBrightPixels(400)[np.newaxis, ...])
        self.assertEqual(w.siValueLabel.text, "0.0000")
        self.assertIn("#4FC3F7", w.siValueLabel.styleSheet)

    def test_StenosisIndexMissingInputs(self):
        self.delayDisplay("Stenosis index waits for both images")
        w = self._createWidget()
        w.expirationSelector.setCurrentNode(self._addVolume(self._imageWithBrightPixels(100), "Exp"))
        self.assertEqual(w.siValueLabel.text, "—")
        self.assertEqual(w.siDetailLabel.text, "Select both images to calculate")

    def test_StenosisIndexZeroInspiration(self):
        self.delayDisplay("Stenosis index is N/A when inspiration area is zero")
        w = self._createWidget()
        w.expirationSelector.setCurrentNode(self._addVolume(self._imageWithBrightPixels(100), "Exp"))
        w.inspirationSelector.setCurrentNode(self._addVolume(self._imageWithBrightPixels(0), "Insp"))
        self.assertEqual(w.siValueLabel.text, "N/A")
        self.assertEqual(w.siDetailLabel.text, "Inspiration pixel count is zero")

    def test_SnapshotConvertsRGBToGrayscale(self):
        self.delayDisplay("Snapshot copies an RGB video frame as grayscale and refreshes depth")
        w = self._createWidget()
        rgb = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 30, 90, 210  # mean = 110
        video = slicer.util.addVolumeFromArray(rgb, name="Video", nodeClassName="vtkMRMLVectorVolumeNode")
        target = self._addVolume(np.zeros((8, 8), np.uint8), "Exp")
        w.bronchVideoSelector.setCurrentNode(video)
        w.expirationSelector.setCurrentNode(target)

        w.onSnapshotExpiration()
        snap = slicer.util.arrayFromVolume(target)
        self.assertEqual(snap.shape, (1, 8, 8))
        self.assertTrue(np.all(snap == 110))
        # Observer on the expiration node regenerates its depth map
        self.assertTrue(np.all(self._array("Exp_Depth") == 110))

        # No target selected -> no-op
        w.inspirationSelector.setCurrentNode(None)
        w.onSnapshotInspiration()

    def test_TrailingSingletonVolumeLayout(self):
        self.delayDisplay("(H, W, 1) volumes are handled like (1, H, W)")
        w = self._createWidget()
        img = self._imageWithBrightPixels(300, shape=(40, 40))
        img[5, 30] = 255  # break symmetry so a transpose would be caught

        # Video and expiration image both arrive as (H, W, 1)
        video = slicer.util.addVolumeFromArray(img[..., np.newaxis], name="Video")
        self.assertEqual(slicer.util.arrayFromVolume(video).shape, (40, 40, 1))
        w.bronchVideoSelector.setCurrentNode(video)
        np.testing.assert_array_equal(self._array("Video_Depth")[0], img)

        exp = slicer.util.addVolumeFromArray(img[..., np.newaxis], name="Exp")
        w.expirationSelector.setCurrentNode(exp)
        np.testing.assert_array_equal(self._array("Exp_Depth")[0], img)

        # Inspiration image in the usual (1, H, W) layout -> SI computes across layouts
        w.inspirationSelector.setCurrentNode(self._addVolume(img, "Insp"))
        self.assertEqual(w.siValueLabel.text, "0.0000")

        # Snapshot from an (H, W, 1) video into the inspiration image
        w.onSnapshotInspiration()
        snap = slicer.util.arrayFromVolume(w.inspirationSelector.currentNode())
        self.assertEqual(snap.shape, (1, 40, 40))
        np.testing.assert_array_equal(snap[0], img)

    def test_BronchVideoDepthUpdatesOnModify(self):
        self.delayDisplay("Bronchoscope depth map follows video updates")
        estimator = _IdentityDepthEstimator()
        w = self._createWidget(estimator)
        video = self._addVolume(np.full((8, 8), 10, np.uint8), "Video")
        w.bronchVideoSelector.setCurrentNode(video)
        self.assertTrue(np.all(self._array("Video_Depth") == 10))
        calls = estimator.calls
        slicer.util.updateVolumeFromArray(video, np.full((1, 8, 8), 50, np.uint8))
        self.assertGreater(estimator.calls, calls)
        self.assertTrue(np.all(self._array("Video_Depth") == 50))

        # Deselecting removes the observer
        w.bronchVideoSelector.setCurrentNode(None)
        calls = estimator.calls
        slicer.util.updateVolumeFromArray(video, np.full((1, 8, 8), 90, np.uint8))
        self.assertEqual(estimator.calls, calls)

    def test_SetupLayout(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            self.delayDisplay("Skipping layout test (no main window)")
            return
        self.delayDisplay("Set Up Layout assigns volumes to slice views")
        w = self._createWidget()
        video = self._addVolume(self._imageWithBrightPixels(100), "Video")
        exp = self._addVolume(self._imageWithBrightPixels(100), "Exp")
        insp = self._addVolume(self._imageWithBrightPixels(400), "Insp")
        w.bronchVideoSelector.setCurrentNode(video)
        w.expirationSelector.setCurrentNode(exp)
        w.inspirationSelector.setCurrentNode(insp)
        w.onSetupLayout()
        self.assertEqual(layoutManager.layout, slicer.vtkMRMLLayoutNode.SlicerLayoutFourByTwoSliceView)

        def composite(name):
            return layoutManager.sliceWidget(name).mrmlSliceCompositeNode()

        expected = {
            "Red": "Video", "Slice5": "Video_Depth",
            "Yellow": "Exp", "Green": "Exp_Depth", "Slice4": "Exp_Threshold",
            "Slice6": "Insp", "Slice7": "Insp_Depth", "Slice8": "Insp_Threshold",
        }
        for sliceName, nodeName in expected.items():
            self.assertEqual(composite(sliceName).GetBackgroundVolumeID(),
                             slicer.util.getNode(nodeName).GetID(), sliceName)
        self.assertEqual(composite("Yellow").GetForegroundVolumeID(), slicer.util.getNode("Exp_Threshold").GetID())
        self.assertEqual(composite("Slice6").GetForegroundVolumeID(), slicer.util.getNode("Insp_Threshold").GetID())

    def test_RealDepthEstimator(self):
        try:
            from StenosisUtils.depth_estimation import DepthEstimator
        except ImportError as e:
            self.delayDisplay(f"Skipping real model test ({e})")
            return
        self.delayDisplay("Real depth estimator produces a 200x200 depth map")
        w = self._createWidget(DepthEstimator())
        yy, xx = np.mgrid[:200, :200]
        img = np.where(np.hypot(yy - 100, xx - 100) < 50, 40, 220).astype(np.uint8)
        depthNode = w.generateDepthMap(self._addVolume(img, "Frame"))
        depth = slicer.util.arrayFromVolume(depthNode)
        self.assertEqual(depthNode.GetName(), "Frame_Depth")
        self.assertEqual(depth.shape, (1, 200, 200))
        self.assertGreater(depth.std(), 0)
