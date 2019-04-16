import os, sys
from shutil import copy2
from paraview.simple import *


name = sys.argv[1]
cluster_home = '/clusterhome/chaudhry'
set_number = '12'

if sys.argv[2] == 'cluster':
    if int(sys.argv[3]) == 1:
        directory = '/scratch/chaudhry'
    if int(sys.argv[3]) == 2: 
        directory = '/net/stzs3/export/scratch2/chaudhry'
    else:
        print('Enter Valid Directory')
if sys.argv[2] == 'local':
    if int(sys.argv[3]) == 1:
        directory = '/usr/people/chaudhry/hydra-scratch'
    if int(sys.argv[3]) == 2:
        directory = '/usr/people/chaudhry/hydra-scratch2'
    else:
        print('Enter Valid Directory')

# # Paths to use when using local laptop:
# cluster_home = '/home/hamza/Research/clusterhome'
# directory = '/home/hamza/Research/hydra-scratch'

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# FileName = ['/scratch/chaudhry/set8/{}/buffersstate_000{:04}0.vtp'.format(name,index) for index in range(1,5000)]
# FileName = ['/usr/people/chaudhry/hydra-scratch/set{}/{}/buffersstate_000{:04}0.vtp'.format(name,index) for index in range(1,5000)]
FileName = ['{}/set{}/{}/buffersstate_000{:03}00.vtp'.format(directory, set_number, name, index) for index in range(1,500)]


# create a new 'XML PolyData Reader'
buffersstate_000 = OpenDataFile(FileName)
buffersstate_000.CellArrayStatus = ['CellType', 'MuscleNumber', 'MuscleActivation']
buffersstate_000.PointArrayStatus = ['ParticleType', 'Velocity']

print 'Data Loaded'

# set active source
SetActiveSource(buffersstate_000)

# get active view
renderView = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView.ViewSize = [1609, 572]

# show data in view
buffersstate_000Display = Show(buffersstate_000, renderView)
# trace defaults for the display properties.
buffersstate_000Display.Representation = 'Surface'
buffersstate_000Display.AmbientColor = [1.0, 1.0, 1.0]
buffersstate_000Display.ColorArrayName = [None, '']
buffersstate_000Display.DiffuseColor = [1.0, 1.0, 1.0]
buffersstate_000Display.LookupTable = None
buffersstate_000Display.MapScalars = 1
buffersstate_000Display.InterpolateScalarsBeforeMapping = 1
buffersstate_000Display.Opacity = 0.8
buffersstate_000Display.PointSize = 2.0
buffersstate_000Display.LineWidth = 1.0
buffersstate_000Display.Interpolation = 'Gouraud'
buffersstate_000Display.Specular = 0.2
buffersstate_000Display.SpecularColor = [1.0, 1.0, 1.0]
buffersstate_000Display.SpecularPower = 100.0
buffersstate_000Display.Ambient = 0.0
buffersstate_000Display.Diffuse = 1.0
buffersstate_000Display.EdgeColor = [0.0, 0.0, 0.5]
buffersstate_000Display.BackfaceRepresentation = 'Follow Frontface'
buffersstate_000Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
buffersstate_000Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
buffersstate_000Display.BackfaceOpacity = 1.0
buffersstate_000Display.Position = [0.0, 0.0, 0.0]
buffersstate_000Display.Scale = [1.0, 1.0, 1.0]
buffersstate_000Display.Orientation = [0.0, 0.0, 0.0]
buffersstate_000Display.Origin = [0.0, 0.0, 0.0]
buffersstate_000Display.Pickable = 1
buffersstate_000Display.Texture = None
buffersstate_000Display.Triangulate = 0
buffersstate_000Display.NonlinearSubdivisionLevel = 1
buffersstate_000Display.OSPRayUseScaleArray = 0
buffersstate_000Display.OSPRayScaleArray = 'ParticleType'
buffersstate_000Display.OSPRayScaleFunction = 'PiecewiseFunction'
buffersstate_000Display.GlyphType = 'Arrow'
buffersstate_000Display.SelectionCellLabelBold = 0
buffersstate_000Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
buffersstate_000Display.SelectionCellLabelFontFamily = 'Arial'
buffersstate_000Display.SelectionCellLabelFontSize = 18
buffersstate_000Display.SelectionCellLabelItalic = 0
buffersstate_000Display.SelectionCellLabelJustification = 'Left'
buffersstate_000Display.SelectionCellLabelOpacity = 1.0
buffersstate_000Display.SelectionCellLabelShadow = 0
buffersstate_000Display.SelectionPointLabelBold = 0
buffersstate_000Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
buffersstate_000Display.SelectionPointLabelFontFamily = 'Arial'
buffersstate_000Display.SelectionPointLabelFontSize = 18
buffersstate_000Display.SelectionPointLabelItalic = 0
buffersstate_000Display.SelectionPointLabelJustification = 'Left'
buffersstate_000Display.SelectionPointLabelOpacity = 1.0
buffersstate_000Display.SelectionPointLabelShadow = 0

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
buffersstate_000Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'Arrow' selected for 'GlyphType'
buffersstate_000Display.GlyphType.TipResolution = 6
buffersstate_000Display.GlyphType.TipRadius = 0.1
buffersstate_000Display.GlyphType.TipLength = 0.35
buffersstate_000Display.GlyphType.ShaftResolution = 6
buffersstate_000Display.GlyphType.ShaftRadius = 0.03
buffersstate_000Display.GlyphType.Invert = 0

# reset view to fit data
renderView.ResetCamera()

# set scalar coloring
ColorBy(buffersstate_000Display, ('CELLS', 'CellType'))

# rescale color and/or opacity maps used to include current data range
buffersstate_000Display.RescaleTransferFunctionToDataRange(True)

# hide color bar/color legend
buffersstate_000Display.SetScalarBarVisibility(renderView, False)

# get color transfer function/color map for 'CellType'
cellTypeLUT = GetColorTransferFunction('CellType')
cellTypeLUT.LockDataRange = 0
cellTypeLUT.InterpretValuesAsCategories = 0
cellTypeLUT.ShowCategoricalColorsinDataRangeOnly = 0
cellTypeLUT.RescaleOnVisibilityChange = 0
cellTypeLUT.EnableOpacityMapping = 1
cellTypeLUT.RGBPoints = [1.0, 0.231373, 0.298039, 0.752941, 1.5, 0.865003, 0.865003, 0.865003, 2.0, 0.8509803921568627, 0.8509803921568627, 0.8509803921568627]
cellTypeLUT.UseLogScale = 0
cellTypeLUT.ColorSpace = 'Diverging'
cellTypeLUT.UseBelowRangeColor = 0
cellTypeLUT.BelowRangeColor = [0.0, 0.0, 0.0]
cellTypeLUT.UseAboveRangeColor = 0
cellTypeLUT.AboveRangeColor = [1.0, 1.0, 1.0]
cellTypeLUT.NanColor = [1.0, 1.0, 0.0]
cellTypeLUT.Discretize = 1
cellTypeLUT.NumberOfTableValues = 256
cellTypeLUT.ScalarRangeInitialized = 1.0
cellTypeLUT.HSVWrap = 0
cellTypeLUT.VectorComponent = 0
cellTypeLUT.VectorMode = 'Magnitude'
cellTypeLUT.AllowDuplicateScalars = 1
cellTypeLUT.Annotations = []
cellTypeLUT.ActiveAnnotatedValues = []
cellTypeLUT.IndexedColors = []

# get opacity transfer function/opacity map for 'CellType'
cellTypePWF = GetOpacityTransferFunction('CellType')
cellTypePWF.Points = [1.0, 0.30263158679008484, 0.5, 0.0, 2.0, 1.0, 0.5, 0.0]
cellTypePWF.AllowDuplicateScalars = 1
cellTypePWF.ScalarRangeInitialized = 1

# Properties modified on renderView.AxesGrid
renderView.AxesGrid.Visibility = 1

# Properties modified on renderView.AxesGrid
renderView.AxesGrid.ZTitle = 'Distance'

# Properties modified on renderView
renderView.OrientationAxesVisibility = 0

#### saving camera placements for all active views

# current camera placement for Vertical
renderView.InteractionMode = '2D'
renderView.CameraPosition = [75.15130925178528, 2148.116489817861, 551.0649273395538]
renderView.CameraFocalPoint = [75.15130925178528, 7.726200073957443, 551.0649273395538]
renderView.CameraViewUp = [1.0, 0.0, 2.220446049250313e-16]
renderView.CameraParallelScale = 176.51311562413633

print 'Display Properties Set'

# If I want the 3D View
# renderView.CameraPosition = [-201.29901463879204, 631.6730450769429, 540.1807036100428]
# renderView.CameraFocalPoint = [75.17673015594482, 8.287799805402756, 551.0693221092224]
# renderView.CameraViewUp = [0.914134044024918, 0.405411331285193, -0.0007758996178430788]
# renderView.CameraParallelScale = 554.0037409695371

# If I want the Entire Track
# renderView.CameraParallelScale = 194.71976008532874

myview = GetActiveView()

print 'Properties Rendered'

# get animation scene
scene = GetAnimationScene()

# current camera placement for Vertical
renderView.InteractionMode = '2D'
renderView.CameraPosition = [75.15130925178528, 2148.116489817861, 551.0649273395538]
renderView.CameraFocalPoint = [75.15130925178528, 7.726200073957443, 551.0649273395538]
renderView.CameraViewUp = [1.0, 0.0, 2.220446049250313e-16]
renderView.CameraParallelScale = 176.51311562413633
renderView.ViewSize = [1550,500]

# Properties modified on scene
scene = GetAnimationScene()
scene.PlayMode = 'Snap To TimeSteps'
scene.GoToFirst()

Show()
Render()

# Create directory to save images in
if not os.path.exists('{}/experiments/images/set{}/{}'.format(cluster_home, set_number, name)):   
    os.mkdir('{}/experiments/images/set{}/{}'.format(cluster_home, set_number, name))

print 'Image Directory Created'

# Save individual images for later rendering
i = 0
while True:
    imageName = "image_%04d.png" % (i)
    i = i + 1
    renderView.Update()
    SaveScreenshot('{}/experiments/images/set{}/{}/{}'.format(cluster_home, set_number, name, imageName), renderView, ImageResolution=[1920,1080])
    if i == scene.EndTime:
        break
    scene.GoToNext()

print 'Images Created'

# Create directory to save videos in
if not os.path.exists('{}/experiments/videos/set{}/{}'.format(cluster_home, set_number, name)):   
    os.mkdir('{}/experiments/videos/set{}/{}'.format(cluster_home, set_number, name))

# # Generate 3 Videos from these Images
# 30 Seconds
os.system("ffmpeg -r 16.6 -f image2 -s 1920x1080 -i {}/experiments/images/set{}/{}/image_%04d.png -vcodec h264 -crf 5 {}/experiments/videos/set{}/{}/30_sec.avi".format(cluster_home, set_number, name, cluster_home, set_number, name))

# 10 Seconds
os.system("ffmpeg -r 49.8 -f image2 -s 1920x1080 -i {}/experiments/images/set{}/{}/image_%04d.png -vcodec h264 -crf 5 {}/experiments/videos/set{}/{}/10_sec.avi".format(cluster_home, set_number, name, cluster_home, set_number, name))

# 1 Second
os.system("ffmpeg -r 498 -f image2 -s 1920x1080 -i {}/experiments/images/set{}/{}/image_%04d.png -vcodec h264 -crf 5 {}/experiments/videos/set{}/{}/01_sec.avi".format(cluster_home, set_number, name, cluster_home, set_number, name))
