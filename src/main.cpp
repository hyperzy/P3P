#include <iostream>
#include <fstream>
#include <camera.h>
#include <opencv2/highgui.hpp>
#include <memory>
#include "algorithms.h"

#include <chrono>
using namespace std;
using namespace cv;

// todo: parameters parser
int main(int argc, char *argv[]) {

    unique_ptr<PnpBase> pnp_module(new PnpBase);
    auto obj_points = pnp_module->loadObjPointsFromFile("../res/reproj_test/obj_points.txt");
    unique_ptr<RGBImage> rgb_cam(new RGBImage);
    rgb_cam->loadImage("../res/calibration000.JPG");
    rgb_cam->loadIntrinsic("../res/intrinsic.txt");
    // choose points by hand
//    auto img_points = pnp_module->selectImgPointsFromImage(rgb_cam.get());
//    pnp_module->saveImgPoints("../res/img_points.txt", img_points);
    // load image points from file
    auto img_points = pnp_module->loadImgPointsFromFile("../res/reproj_test/img_points.txt");
//    pnp_module->rectP4P(rgb_cam.get(), obj_points, img_points, 2);
    unique_ptr<P3P> p3p(new P3P);
    p3p->solve(rgb_cam.get(), obj_points, img_points);
    return 0;
    // execution time test
    /*
    execDuration(pnp_module.get(), rgb_cam.get(), obj_points, img_points, 0);
    cout << pnp_module->reprojectError(rgb_cam.get(), obj_points, img_points) << endl;
    execDuration(pnp_module.get(), rgb_cam.get(), obj_points, img_points, 2);
    cout << pnp_module->reprojectError(rgb_cam.get(), obj_points, img_points) << endl;
    img_points = pnp_module->loadImgPointsFromFile("../res/reproj_test/img_points_RH.txt");
    obj_points = pnp_module->loadObjPointsFromFile("../res/reproj_test/obj_points_RH.txt");
    execDuration(pnp_module.get(), rgb_cam.get(), obj_points, img_points, 1);
    cout << pnp_module->reprojectError(rgb_cam.get(), obj_points, img_points) << endl;

     reprojection error test
    pnp_module->noiseTest(rgb_cam.get(), "../res/reproj_test/obj_points_err_test.txt", "../res/reproj_test/img_points_err_test.txt",
                        "../res/reproj_test/test_obj_points.txt", "../res/reproj_test/test_img_points.txt");
     */
    Mat3 R;
    R << 0.697684341864769, -0.715726403617393, 0.0311813130197743,
        -0.480146037816071, -0.499458394758601, -0.72111101383538,
         0.53169196108861, 0.488136279198826, -0.692117498293226;
    Vec3 t(.003737, -.004009, .720708);
    rgb_cam->setRotation(R);
    rgb_cam->setTranslation(t);
    pnp_module->noiseTeset(rgb_cam.get(), 3.);
    return 0;

    ofstream fout("../res/rotation.mat", ofstream::out);
    rgb_cam->saveRotation(fout);
    fout.close();
    fout.open("../res/translation.mat", ofstream::out);
    rgb_cam->saveTranslation(fout);
    fout.close();

    return 0;
}
