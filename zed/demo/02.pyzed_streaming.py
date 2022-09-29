import sys 
sys.path.append("..")
import utils.zed_viewer as gl
import pyzed.sl as sl

if __name__ == "__main__":
    print("Running Depth Sensing sample ... Press 'Esc' to quit")

    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 640#720
    res.height = 480#404

    camera_model = zed.get_camera_information().camera_model
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(len(sys.argv), sys.argv, camera_model, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            viewer.updateData(point_cloud)

    viewer.exit()
    zed.close()
